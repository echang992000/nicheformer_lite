"""
Nicheformer Knowledge-Distillation Trainer.

Orchestrates the full distillation pipeline:

  1. Optionally fit a :class:`GeneModuleTokenizer` on training data.
  2. Generate BERT-style masked inputs for the MLM objective.
  3. Run the teacher in eval / no-grad mode to produce soft targets
     and intermediate representations.
  4. Train the student with the combined distillation loss.
  5. Checkpoint and evaluate periodically.

Features:
  * tqdm progress bar with live loss display
  * Early stopping with configurable patience
  * Mixed-precision training (AMP) on CUDA
  * Full checkpoint resume (optimizer, scheduler, step state)
  * CPU, CUDA, and Apple-Silicon MPS backends
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm.auto import tqdm
except ImportError:  # graceful fallback
    tqdm = None  # type: ignore[assignment]

from ..config import DistillationConfig, NicheformerConfig, StudentConfig
from ..models.teacher import NicheformerTeacher
from ..models.student import NicheformerStudent
from ..models.tokenizer import GeneModuleTokenizer
from .losses import CombinedDistillationLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLM masking (follows Nicheformer / BERT protocol)
# ---------------------------------------------------------------------------

def apply_mlm_masking(
    input_ids: torch.Tensor,
    masking_p: float = 0.15,
    n_tokens: int = 20340,
    aux_tokens: int = 30,
    pad_token: int = 1,
    mask_token: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply BERT-style masking to tokenised input.

    Parameters
    ----------
    input_ids : (B, S)  int64 token tensor.
    masking_p : probability of masking each eligible token.
    n_tokens  : gene vocabulary size (excluding aux tokens).
    aux_tokens: number of reserved special-token indices.
    pad_token : index used for padding.
    mask_token: index used for the [MASK] token.

    Returns
    -------
    masked_input : same shape, with masked positions replaced.
    labels       : ground-truth token ids; -100 at unmasked positions.
    mask_bool    : bool tensor, True at masked positions.

    Raises
    ------
    ValueError
        If ``input_ids`` is not a 2-D tensor.
    """
    if input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must be 2-D (batch, seq_len), got shape {input_ids.shape}"
        )

    labels = input_ids.clone()
    masked_input = input_ids.clone()

    # Determine maskable positions (not PAD, not special tokens < aux_tokens)
    maskable = (input_ids >= aux_tokens) & (input_ids != pad_token)
    prob_matrix = torch.full_like(input_ids, masking_p, dtype=torch.float)
    prob_matrix[~maskable] = 0.0

    mask_bool = torch.bernoulli(prob_matrix).bool()
    labels[~mask_bool] = -100

    # 80 % MASK, 10 % random, 10 % keep
    indices_replaced = torch.bernoulli(
        torch.full_like(prob_matrix, 0.8)
    ).bool() & mask_bool
    masked_input[indices_replaced] = mask_token

    indices_random = (
        torch.bernoulli(torch.full_like(prob_matrix, 0.5)).bool()
        & mask_bool
        & ~indices_replaced
    )
    random_tokens = torch.randint(
        aux_tokens, n_tokens + aux_tokens,
        size=input_ids.shape, device=input_ids.device,
    )
    masked_input[indices_random] = random_tokens[indices_random]

    return masked_input, labels, mask_bool


# ---------------------------------------------------------------------------
# Synthetic data generator (for testing / demo)
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_cells: int = 5000,
    n_tokens: int = 20340,
    seq_len: int = 512,
    aux_tokens: int = 30,
) -> torch.Tensor:
    """Generate rank-tokenised synthetic single-cell data.

    Each cell gets a random subset of genes sorted by a synthetic
    expression profile, mimicking the Nicheformer tokenisation.
    """
    tokens = torch.zeros(n_cells, seq_len, dtype=torch.long)
    for i in range(n_cells):
        n_expressed = np.random.randint(50, min(seq_len, n_tokens))
        gene_indices = np.random.choice(n_tokens, n_expressed, replace=False)
        expr = np.random.exponential(1.0, size=n_expressed)
        order = np.argsort(-expr)
        sorted_genes = gene_indices[order][:seq_len]
        tokens[i, : len(sorted_genes)] = torch.from_numpy(sorted_genes + aux_tokens)
    return tokens


# ---------------------------------------------------------------------------
# Distiller
# ---------------------------------------------------------------------------

class NicheformerDistiller:
    """End-to-end knowledge-distillation trainer.

    Parameters
    ----------
    teacher : NicheformerTeacher
    student : NicheformerStudent
    config  : DistillationConfig
    tokenizer : GeneModuleTokenizer, optional
        If the student uses gene-module tokens, the tokenizer is applied
        to teacher inputs before feeding the student.
    """

    def __init__(
        self,
        teacher: NicheformerTeacher,
        student: NicheformerStudent,
        config: Optional[DistillationConfig] = None,
        tokenizer: Optional[GeneModuleTokenizer] = None,
    ):
        if config is None:
            config = DistillationConfig()
        self.config = config
        self.tokenizer = tokenizer

        # Device — validate availability
        self.device = self._resolve_device(config.device)
        self.teacher = teacher.to(self.device).eval()
        self.student = student.to(self.device)

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Layer mapping: default = evenly spaced
        layer_map = config.layer_mapping
        if layer_map is None:
            s_layers = student.config.nlayers
            t_layers = teacher.config.nlayers
            stride = max(1, t_layers // s_layers)
            layer_map = {
                i: min(i * stride + stride, t_layers)
                for i in range(s_layers)
            }
        self.layer_mapping = layer_map

        # Hidden-state projection
        if student.config.dim_model != teacher.config.dim_model:
            student.set_teacher_projection(teacher.config.dim_model)
            student = student.to(self.device)

        # Loss
        self.criterion = CombinedDistillationLoss(
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            delta=config.delta,
            temperature=config.temperature,
            layer_mapping=self.layer_mapping,
            student_dim=student.config.dim_model,
            teacher_dim=teacher.config.dim_model,
        ).to(self.device)

        # Optimiser
        params = list(self.student.parameters()) + list(self.criterion.parameters())
        self.optimizer = torch.optim.AdamW(
            params, lr=config.learning_rate, weight_decay=config.weight_decay,
        )

        # LR schedule: linear warmup + cosine decay
        def lr_lambda(step: int) -> float:
            if step < config.warmup_steps:
                return step / max(1, config.warmup_steps)
            progress = (step - config.warmup_steps) / max(
                1, config.max_steps - config.warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda,
        )

        # AMP scaler (CUDA only)
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # State
        self.global_step = 0
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.history: List[Dict[str, Any]] = []

    # --------------------------------------------------------- device helper

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Validate and return a :class:`torch.device`, falling back to CPU."""
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable — falling back to CPU.")
            return torch.device("cpu")
        if device == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            logger.warning("MPS requested but unavailable — falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device)

    # ---------------------------------------------------------------- train

    def train(
        self,
        dataloader: DataLoader,
        max_steps: Optional[int] = None,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, list]:
        """Run the distillation training loop.

        Parameters
        ----------
        dataloader : training data (batches of token tensors).
        max_steps  : override ``config.max_steps``.
        eval_dataloader : optional validation set.

        Returns
        -------
        history : dict with loss curves.
        """
        max_steps = max_steps or self.config.max_steps
        cfg = self.config
        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.student.train()
        accum_loss: Dict[str, float] = {}
        accum_n = 0
        t0 = time.time()

        logger.info(
            "Starting distillation: %d steps, lr=%.2e, device=%s, amp=%s",
            max_steps, cfg.learning_rate, self.device, self.use_amp,
        )

        # Progress bar
        pbar = None
        if tqdm is not None:
            pbar = tqdm(
                total=max_steps,
                initial=self.global_step,
                desc="Distilling",
                unit="step",
            )

        stopped_early = False

        while self.global_step < max_steps:
            for batch in dataloader:
                if self.global_step >= max_steps:
                    break

                loss, components = self._train_step(batch)

                # Accumulate for logging
                for k, v in components.items():
                    accum_loss[k] = accum_loss.get(k, 0.0) + v
                accum_n += 1

                # Gradient accumulation + AMP
                if (self.global_step + 1) % cfg.gradient_accumulation == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.student.parameters(), cfg.max_grad_norm,
                    )
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                # Progress bar update
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        loss=f"{components.get('total_loss', 0):.4f}",
                        mlm=f"{components.get('mlm_loss', 0):.4f}",
                    )

                # Logging
                if (self.global_step + 1) % cfg.log_every == 0 and accum_n > 0:
                    avg = {k: v / accum_n for k, v in accum_loss.items()}
                    elapsed = time.time() - t0
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        "step %d | loss %.4f | mlm %.4f | lr %.2e | %.1f s",
                        self.global_step + 1,
                        avg.get("total_loss", 0),
                        avg.get("mlm_loss", 0),
                        lr, elapsed,
                    )
                    self.history.append({
                        "step": self.global_step + 1,
                        **avg,
                        "lr": lr,
                    })
                    accum_loss.clear()
                    accum_n = 0

                # Evaluation
                if eval_dataloader and (self.global_step + 1) % cfg.eval_every == 0:
                    eval_loss = self._evaluate(eval_dataloader)
                    logger.info(
                        "step %d | eval_loss %.4f",
                        self.global_step + 1, eval_loss,
                    )
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.patience_counter = 0
                        self.student.save(str(ckpt_dir / "best_student.pt"))
                    else:
                        self.patience_counter += 1

                    # Early stopping
                    if cfg.early_stopping and self.patience_counter >= cfg.patience:
                        logger.info(
                            "Early stopping at step %d (patience=%d exhausted).",
                            self.global_step + 1, cfg.patience,
                        )
                        stopped_early = True
                        break

                # Checkpoint
                if (self.global_step + 1) % cfg.save_every == 0:
                    self.student.save(
                        str(ckpt_dir / f"student_step_{self.global_step + 1}.pt")
                    )

                self.global_step += 1

            if stopped_early:
                break

        if pbar is not None:
            pbar.close()

        # Final save
        self.student.save(str(ckpt_dir / "final_student.pt"))
        logger.info("Distillation complete after %d steps.", self.global_step)
        return {"history": self.history}

    # ------------------------------------------------------------- internals

    def _train_step(
        self, batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single training step."""
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        input_ids = batch.to(self.device)

        teacher_config = self.teacher.config
        student_config = self.student.config

        # Apply MLM masking for teacher
        masked_teacher, labels, mask_bool = apply_mlm_masking(
            input_ids,
            masking_p=teacher_config.masking_p,
            n_tokens=teacher_config.n_tokens,
            aux_tokens=teacher_config.aux_tokens,
        )
        padding_mask = (input_ids == 1) | (input_ids == 0)

        need_hidden = self.config.gamma > 0
        need_attn = self.config.delta > 0

        # Teacher forward (always full precision)
        with torch.no_grad():
            teacher_out = self.teacher(
                masked_teacher,
                padding_mask=padding_mask,
                output_hidden_states=need_hidden,
                output_attentions=need_attn,
            )

        # Student forward (with optional AMP)
        student_input = masked_teacher

        autocast_ctx = (
            torch.cuda.amp.autocast(enabled=True)
            if self.use_amp
            else _nullcontext()
        )
        with autocast_ctx:
            student_out = self.student(
                student_input,
                padding_mask=padding_mask,
                output_hidden_states=need_hidden,
                output_attentions=need_attn,
            )
            loss, components = self.criterion(
                student_out, teacher_out, labels, masked_positions=mask_bool,
            )

        # Scale for gradient accumulation
        scaled_loss = loss / self.config.gradient_accumulation
        if self.use_amp:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return loss, components

    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader) -> float:
        self.student.eval()
        total_loss = 0.0
        n = 0
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            input_ids = batch.to(self.device)

            masked, labels, mask_bool = apply_mlm_masking(
                input_ids,
                masking_p=self.teacher.config.masking_p,
                n_tokens=self.teacher.config.n_tokens,
                aux_tokens=self.teacher.config.aux_tokens,
            )
            padding_mask = (input_ids == 1) | (input_ids == 0)

            teacher_out = self.teacher(masked, padding_mask=padding_mask)
            student_out = self.student(masked, padding_mask=padding_mask)

            loss, _ = self.criterion(
                student_out, teacher_out, labels, masked_positions=mask_bool,
            )
            total_loss += loss.item() * input_ids.size(0)
            n += input_ids.size(0)

        self.student.train()
        return total_loss / max(n, 1)

    # -------------------------------------------------- checkpoint resume

    def save_checkpoint(self, path: str) -> None:
        """Save full training state for resuming later.

        Saves the student weights, optimizer, scheduler, scaler, and
        step counters so that training can be resumed exactly where
        it left off.
        """
        torch.save({
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "patience_counter": self.patience_counter,
            "student_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "criterion_state_dict": self.criterion.state_dict(),
            "history": self.history,
            "config": self.config,
        }, path)
        logger.info("Full checkpoint saved to %s (step %d)", path, self.global_step)

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a full checkpoint.

        Parameters
        ----------
        path : str
            Path to a checkpoint saved by :meth:`save_checkpoint`.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.student.load_state_dict(ckpt["student_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if self.use_amp and ckpt.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.criterion.load_state_dict(ckpt["criterion_state_dict"])
        self.global_step = ckpt["global_step"]
        self.best_loss = ckpt["best_loss"]
        self.patience_counter = ckpt.get("patience_counter", 0)
        self.history = ckpt.get("history", [])
        logger.info("Resumed from %s at step %d", path, self.global_step)

    # -------------------------------------------------- convenience builders

    @classmethod
    def from_configs(
        cls,
        teacher_config: Optional[NicheformerConfig] = None,
        student_config: Optional[StudentConfig] = None,
        distill_config: Optional[DistillationConfig] = None,
        teacher_checkpoint: Optional[str] = None,
    ) -> "NicheformerDistiller":
        """Construct a distiller from configuration objects.

        Parameters
        ----------
        teacher_checkpoint : str, optional
            Path to a pretrained teacher ``.ckpt``.  If ``None``, a
            randomly initialised teacher is used (useful for testing).
        """
        if teacher_checkpoint:
            teacher = NicheformerTeacher.from_pretrained(
                teacher_checkpoint, config=teacher_config,
            )
        else:
            teacher = NicheformerTeacher(teacher_config)

        student = NicheformerStudent(student_config)

        return cls(teacher, student, distill_config)

    @staticmethod
    def make_dataloader(
        tokens: torch.Tensor, batch_size: int = 32, shuffle: bool = True,
    ) -> DataLoader:
        """Wrap a token tensor in a DataLoader."""
        ds = TensorDataset(tokens)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)


# ---------------------------------------------------------------------------
# Tiny helper for optional context managers
# ---------------------------------------------------------------------------

class _nullcontext:
    """Minimal no-op context manager (for Python 3.9 compat)."""
    def __enter__(self) -> None:
        return None
    def __exit__(self, *args: object) -> None:
        pass
