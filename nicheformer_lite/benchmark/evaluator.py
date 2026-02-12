"""
Benchmark suite for comparing original Nicheformer and Nicheformer-Lite.

Measures three axes:

  **Size**     Parameter count, memory footprint (float32 / float16).
  **Speed**    Inference latency and throughput across batch sizes and
               sequence lengths.
  **Quality**  MLM accuracy, top-k recall, per-token perplexity, and
               Centered Kernel Alignment (CKA) between teacher and
               student embeddings.

All results are collected into a :class:`BenchmarkReport` that can be
printed, serialised to JSON, or visualised with matplotlib.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import BenchmarkConfig
from ..distillation.distiller import apply_mlm_masking, generate_synthetic_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkReport:
    """Container for benchmark results."""
    teacher_name: str = "NicheformerTeacher"
    student_name: str = "NicheformerStudent"

    # Size
    teacher_params: int = 0
    student_params: int = 0
    teacher_memory_mb: float = 0.0
    student_memory_mb: float = 0.0
    compression_ratio: float = 0.0

    # Speed  {(batch_size, seq_len): latency_ms}
    teacher_latency: Dict[str, float] = field(default_factory=dict)
    student_latency: Dict[str, float] = field(default_factory=dict)
    speedup: Dict[str, float] = field(default_factory=dict)

    # Quality
    teacher_mlm_accuracy: float = 0.0
    student_mlm_accuracy: float = 0.0
    teacher_top5_recall: float = 0.0
    student_top5_recall: float = 0.0
    teacher_perplexity: float = 0.0
    student_perplexity: float = 0.0
    cka_similarity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "size": {
                "teacher_params": self.teacher_params,
                "student_params": self.student_params,
                "teacher_memory_mb": round(self.teacher_memory_mb, 2),
                "student_memory_mb": round(self.student_memory_mb, 2),
                "compression_ratio": round(self.compression_ratio, 2),
            },
            "speed": {
                "teacher_latency_ms": self.teacher_latency,
                "student_latency_ms": self.student_latency,
                "speedup": self.speedup,
            },
            "quality": {
                "teacher_mlm_accuracy": round(self.teacher_mlm_accuracy, 4),
                "student_mlm_accuracy": round(self.student_mlm_accuracy, 4),
                "teacher_top5_recall": round(self.teacher_top5_recall, 4),
                "student_top5_recall": round(self.student_top5_recall, 4),
                "teacher_perplexity": round(self.teacher_perplexity, 2),
                "student_perplexity": round(self.student_perplexity, 2),
                "cka_similarity": round(self.cka_similarity, 4),
            },
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "",
            "=" * 70,
            "  NICHEFORMER vs NICHEFORMER-LITE  BENCHMARK REPORT",
            "=" * 70,
            "",
            "  MODEL SIZE",
            "  " + "-" * 50,
            f"  Teacher parameters:  {self.teacher_params:>12,}",
            f"  Student parameters:  {self.student_params:>12,}",
            f"  Compression ratio:   {self.compression_ratio:>12.1f}x",
            f"  Teacher memory:      {self.teacher_memory_mb:>12.1f} MB",
            f"  Student memory:      {self.student_memory_mb:>12.1f} MB",
            "",
            "  INFERENCE SPEED",
            "  " + "-" * 50,
        ]
        for key in sorted(self.teacher_latency.keys()):
            t_lat = self.teacher_latency[key]
            s_lat = self.student_latency.get(key, 0)
            spd = self.speedup.get(key, 0)
            lines.append(
                f"  {key:<20s}  teacher {t_lat:>7.1f} ms  "
                f"student {s_lat:>7.1f} ms  ({spd:.1f}x)"
            )
        lines += [
            "",
            "  PREDICTION QUALITY",
            "  " + "-" * 50,
            f"  MLM accuracy  teacher: {self.teacher_mlm_accuracy:.4f}"
            f"   student: {self.student_mlm_accuracy:.4f}",
            f"  Top-5 recall  teacher: {self.teacher_top5_recall:.4f}"
            f"   student: {self.student_top5_recall:.4f}",
            f"  Perplexity    teacher: {self.teacher_perplexity:.2f}"
            f"   student: {self.student_perplexity:.2f}",
            f"  CKA similarity (teacher-student): {self.cka_similarity:.4f}",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _mlm_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy on masked positions only."""
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    correct = (preds == labels) & mask
    return (correct.sum().float() / mask.sum().float()).item()


def _topk_recall(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Top-k recall on masked positions."""
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    topk = logits.topk(k, dim=-1).indices  # (B, S, k)
    labels_exp = labels.unsqueeze(-1).expand_as(topk)
    hits = (topk == labels_exp).any(dim=-1) & mask
    return (hits.sum().float() / mask.sum().float()).item()


def _perplexity(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Per-token perplexity on masked positions."""
    mask = labels != -100
    if mask.sum() == 0:
        return float("inf")
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather log-prob of the true label
    labels_clamped = labels.clamp(min=0)
    token_log_probs = log_probs.gather(-1, labels_clamped.unsqueeze(-1)).squeeze(-1)
    masked_log_probs = token_log_probs[mask]
    avg_nll = -masked_log_probs.mean().item()
    return min(float(np.exp(avg_nll)), 1e6)


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two representation matrices.

    Kornblith et al., 2019.  X, Y have shape (n_samples, dim).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    # Replace non-finite values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2

    denom = np.sqrt(hsic_xx * hsic_yy)
    if not np.isfinite(denom) or denom < 1e-12:
        return 0.0
    result = float(hsic_xy / denom)
    return result if np.isfinite(result) else 0.0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class NicheformerBenchmark:
    """Comprehensive benchmark comparing teacher and student models.

    Parameters
    ----------
    teacher : nn.Module
        The full Nicheformer model.
    student : nn.Module
        The distilled / reduced model.
    config  : BenchmarkConfig
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[BenchmarkConfig] = None,
    ):
        if config is None:
            config = BenchmarkConfig()
        self.config = config
        self.device = torch.device(config.device)
        self.teacher = teacher.to(self.device).eval()
        self.student = student.to(self.device).eval()
        self.report = BenchmarkReport()

    # ------------------------------------------------------------------ size

    def benchmark_size(self) -> Dict[str, float]:
        """Compare parameter counts and memory footprint."""
        t_params = sum(p.numel() for p in self.teacher.parameters())
        s_params = sum(p.numel() for p in self.student.parameters())

        self.report.teacher_params = t_params
        self.report.student_params = s_params
        self.report.compression_ratio = t_params / max(s_params, 1)
        self.report.teacher_memory_mb = t_params * 4 / 1e6
        self.report.student_memory_mb = s_params * 4 / 1e6

        logger.info(
            "Size: teacher=%d, student=%d, ratio=%.1fx",
            t_params, s_params, self.report.compression_ratio,
        )
        return {
            "teacher_params": t_params,
            "student_params": s_params,
            "compression_ratio": self.report.compression_ratio,
        }

    # ----------------------------------------------------------------- speed

    def benchmark_speed(self) -> Dict[str, Dict[str, float]]:
        """Measure inference latency across batch sizes and sequence lengths."""
        cfg = self.config

        for bs in cfg.batch_sizes:
            for sl in cfg.seq_lengths:
                key = f"bs{bs}_seq{sl}"
                dummy = torch.randint(
                    30, 20340, (bs, sl), device=self.device,
                )
                t_lat = self._measure_latency(self.teacher, dummy)
                s_lat = self._measure_latency(self.student, dummy)
                spd = t_lat / max(s_lat, 1e-6)

                self.report.teacher_latency[key] = round(t_lat, 2)
                self.report.student_latency[key] = round(s_lat, 2)
                self.report.speedup[key] = round(spd, 2)

                logger.info(
                    "Speed [%s]: teacher %.1f ms, student %.1f ms (%.1fx)",
                    key, t_lat, s_lat, spd,
                )

        return {
            "teacher": self.report.teacher_latency,
            "student": self.report.student_latency,
            "speedup": self.report.speedup,
        }

    @torch.no_grad()
    def _measure_latency(self, model: nn.Module, x: torch.Tensor) -> float:
        """Return average inference time in milliseconds."""
        cfg = self.config
        # Warmup
        for _ in range(cfg.n_warmup_runs):
            model(x)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(cfg.n_benchmark_runs):
            t0 = time.perf_counter()
            model(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        return float(np.median(times))

    # --------------------------------------------------------------- quality

    @torch.no_grad()
    def benchmark_quality(
        self,
        data: Optional[torch.Tensor] = None,
        n_cells: int = 2000,
    ) -> Dict[str, float]:
        """Evaluate MLM accuracy, top-5 recall, perplexity, and CKA.

        Parameters
        ----------
        data : (n_cells, seq_len) token tensor.
            If ``None``, synthetic data is generated.
        """
        if data is None:
            n_tokens = getattr(self.teacher, "config", None)
            n_tok = n_tokens.n_tokens if n_tokens else 20340
            data = generate_synthetic_data(
                n_cells=n_cells, n_tokens=n_tok, seq_len=512,
            )

        data = data.to(self.device)
        teacher_cfg = getattr(self.teacher, "config", None)
        n_tokens = teacher_cfg.n_tokens if teacher_cfg else 20340
        aux = teacher_cfg.aux_tokens if teacher_cfg else 30
        masking_p = teacher_cfg.masking_p if teacher_cfg else 0.15

        # Process in mini-batches
        bs = min(64, data.size(0))
        t_accs, s_accs = [], []
        t_top5, s_top5 = [], []
        t_ppls, s_ppls = [], []
        t_embeds, s_embeds = [], []

        for i in range(0, data.size(0), bs):
            batch = data[i : i + bs]
            masked, labels, mask_bool = apply_mlm_masking(
                batch, masking_p=masking_p, n_tokens=n_tokens, aux_tokens=aux,
            )
            pad_mask = (batch == 1) | (batch == 0)

            t_out = self.teacher(
                masked, padding_mask=pad_mask, output_hidden_states=True,
            )
            s_out = self.student(
                masked, padding_mask=pad_mask, output_hidden_states=True,
            )

            t_logits = t_out["mlm_logits"]
            s_logits = s_out["mlm_logits"]

            # Accuracy / recall (only when vocab sizes match)
            if t_logits.size(-1) == s_logits.size(-1):
                t_accs.append(_mlm_accuracy(t_logits, labels))
                s_accs.append(_mlm_accuracy(s_logits, labels))
                t_top5.append(_topk_recall(t_logits, labels, k=5))
                s_top5.append(_topk_recall(s_logits, labels, k=5))
                t_ppls.append(_perplexity(t_logits, labels))
                s_ppls.append(_perplexity(s_logits, labels))
            else:
                # Different vocab => only measure student
                s_accs.append(0.0)
                s_top5.append(0.0)
                s_ppls.append(0.0)

            # Embeddings for CKA (last hidden, mean-pooled)
            t_h = t_out["hidden_states"][-1].mean(dim=1).cpu().numpy()
            s_h = s_out["hidden_states"][-1].mean(dim=1).cpu().numpy()
            t_embeds.append(t_h)
            s_embeds.append(s_h)

        self.report.teacher_mlm_accuracy = float(np.mean(t_accs)) if t_accs else 0.0
        self.report.student_mlm_accuracy = float(np.mean(s_accs)) if s_accs else 0.0
        self.report.teacher_top5_recall = float(np.mean(t_top5)) if t_top5 else 0.0
        self.report.student_top5_recall = float(np.mean(s_top5)) if s_top5 else 0.0
        self.report.teacher_perplexity = float(np.mean(t_ppls)) if t_ppls else 0.0
        self.report.student_perplexity = float(np.mean(s_ppls)) if s_ppls else 0.0

        # CKA between teacher and student embeddings
        T = np.concatenate(t_embeds, axis=0)
        S = np.concatenate(s_embeds, axis=0)
        # Truncate to common number of features or use concatenated version
        self.report.cka_similarity = _linear_cka(T, S)

        logger.info(
            "Quality: teacher_acc=%.4f, student_acc=%.4f, CKA=%.4f",
            self.report.teacher_mlm_accuracy,
            self.report.student_mlm_accuracy,
            self.report.cka_similarity,
        )
        return {
            "teacher_mlm_accuracy": self.report.teacher_mlm_accuracy,
            "student_mlm_accuracy": self.report.student_mlm_accuracy,
            "cka_similarity": self.report.cka_similarity,
        }

    # --------------------------------------------------------------- full run

    def run_all(
        self,
        data: Optional[torch.Tensor] = None,
    ) -> BenchmarkReport:
        """Run the full benchmark suite (size + speed + quality).

        Returns the completed :class:`BenchmarkReport`.
        """
        logger.info("Running full benchmark suite...")
        self.benchmark_size()
        self.benchmark_speed()
        self.benchmark_quality(data)
        return self.report

    # ------------------------------------------------------------- plotting

    def plot_results(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> None:
        """Generate comparison plots and save to ``output_dir``.

        Creates three figures:
          * ``size_comparison.png``   – parameter and memory bar chart
          * ``speed_comparison.png``  – latency grouped bar chart
          * ``quality_comparison.png``– accuracy / perplexity radar chart
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed; skipping plots.")
            return

        if output_dir is None:
            output_dir = self.config.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        r = self.report

        # ---- 1. Size comparison ----
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Parameters
        ax = axes[0]
        models = ["Teacher", "Student"]
        params = [r.teacher_params / 1e6, r.student_params / 1e6]
        bars = ax.bar(models, params, color=["#2196F3", "#4CAF50"], width=0.5)
        ax.set_ylabel("Parameters (millions)")
        ax.set_title("Parameter Count")
        for bar, val in zip(bars, params):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}M", ha="center", fontsize=10,
            )

        # Memory
        ax = axes[1]
        mem = [r.teacher_memory_mb, r.student_memory_mb]
        bars = ax.bar(models, mem, color=["#2196F3", "#4CAF50"], width=0.5)
        ax.set_ylabel("Memory (MB, float32)")
        ax.set_title(f"Memory Footprint  ({r.compression_ratio:.1f}x compression)")
        for bar, val in zip(bars, mem):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", fontsize=10,
            )

        plt.tight_layout()
        fig.savefig(output_dir / "size_comparison.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        # ---- 2. Speed comparison ----
        if r.teacher_latency:
            keys = sorted(r.teacher_latency.keys())
            t_vals = [r.teacher_latency[k] for k in keys]
            s_vals = [r.student_latency[k] for k in keys]
            x = np.arange(len(keys))
            width = 0.35

            fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.5), 5))
            ax.bar(x - width / 2, t_vals, width, label="Teacher", color="#2196F3")
            ax.bar(x + width / 2, s_vals, width, label="Student", color="#4CAF50")
            ax.set_xticks(x)
            ax.set_xticklabels(keys, rotation=45, ha="right")
            ax.set_ylabel("Latency (ms)")
            ax.set_title("Inference Latency Comparison")
            ax.legend()

            # Add speedup labels
            for i, k in enumerate(keys):
                spd = r.speedup.get(k, 0)
                ax.text(
                    i, max(t_vals[i], s_vals[i]) + 1,
                    f"{spd:.1f}x", ha="center", fontsize=9, color="#666",
                )

            plt.tight_layout()
            fig.savefig(
                output_dir / "speed_comparison.png", dpi=150, bbox_inches="tight",
            )
            if show:
                plt.show()
            plt.close(fig)

        # ---- 3. Quality comparison ----
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Accuracy
        ax = axes[0]
        vals = [r.teacher_mlm_accuracy, r.student_mlm_accuracy]
        bars = ax.bar(models, vals, color=["#2196F3", "#4CAF50"], width=0.5)
        ax.set_ylabel("Accuracy")
        ax.set_title("MLM Top-1 Accuracy")
        ax.set_ylim(0, max(max(vals) * 1.3, 0.1))
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=10,
            )

        # Top-5 Recall
        ax = axes[1]
        vals = [r.teacher_top5_recall, r.student_top5_recall]
        bars = ax.bar(models, vals, color=["#2196F3", "#4CAF50"], width=0.5)
        ax.set_ylabel("Recall@5")
        ax.set_title("MLM Top-5 Recall")
        ax.set_ylim(0, max(max(vals) * 1.3, 0.1))
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=10,
            )

        # CKA
        ax = axes[2]
        ax.bar(["CKA"], [r.cka_similarity], color="#FF9800", width=0.4)
        ax.set_ylabel("Linear CKA")
        ax.set_title("Representation Similarity")
        ax.set_ylim(0, 1.05)
        ax.text(0, r.cka_similarity + 0.02, f"{r.cka_similarity:.4f}",
                ha="center", fontsize=10)

        plt.tight_layout()
        fig.savefig(
            output_dir / "quality_comparison.png", dpi=150, bbox_inches="tight",
        )
        if show:
            plt.show()
        plt.close(fig)

        logger.info("Plots saved to %s", output_dir)
