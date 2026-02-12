"""
Knowledge-distillation loss functions for Nicheformer-Lite.

Three complementary loss signals are combined:

  **Soft-target loss** (Hinton et al., 2015)
      KL divergence between temperature-scaled teacher and student logits.
      Transfers the teacher's "dark knowledge" — the relative probabilities
      it assigns to non-target tokens.

  **Hidden-state loss**
      Mean-squared error between aligned hidden representations of mapped
      teacher / student layers.  Requires a dimension-projection layer
      when the teacher and student widths differ.

  **Attention-transfer loss** (Zagoruyko & Komodakis, 2017)
      Frobenius-norm distance between teacher and student attention maps,
      averaged across heads and layers.  Encourages the student to attend
      to the same positional relationships.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual losses
# ---------------------------------------------------------------------------

def soft_target_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
    ignore_index: int = -100,
    label_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """KL divergence on temperature-scaled softmax distributions.

    Parameters
    ----------
    student_logits, teacher_logits : (B, S, V)
    temperature : float
    label_mask : (B, S) bool, optional
        If provided, loss is computed only at ``True`` positions
        (e.g. masked tokens in MLM).

    Returns
    -------
    scalar loss
    """
    s_log = F.log_softmax(student_logits / temperature, dim=-1)
    t_prob = F.softmax(teacher_logits / temperature, dim=-1)

    # Per-position KL
    kl = F.kl_div(s_log, t_prob, reduction="none").sum(dim=-1)  # (B, S)

    if label_mask is not None:
        kl = kl * label_mask.float()
        n = label_mask.float().sum().clamp(min=1)
        return (kl.sum() / n) * (temperature ** 2)

    return kl.mean() * (temperature ** 2)


def hidden_state_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    projection: Optional[nn.Linear] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """MSE between aligned hidden representations.

    Parameters
    ----------
    student_hidden, teacher_hidden : (B, S, D_s) / (B, S, D_t)
    projection : nn.Linear, optional
        Maps student dim -> teacher dim when they differ.
    normalize : bool
        Layer-normalise both before comparison (more stable).
    """
    if projection is not None:
        student_hidden = projection(student_hidden)

    if normalize:
        student_hidden = F.layer_norm(
            student_hidden, student_hidden.shape[-1:]
        )
        teacher_hidden = F.layer_norm(
            teacher_hidden, teacher_hidden.shape[-1:]
        )

    return F.mse_loss(student_hidden, teacher_hidden)


def attention_transfer_loss(
    student_attn: torch.Tensor,
    teacher_attn: torch.Tensor,
) -> torch.Tensor:
    """Attention-map transfer via MSE on head-averaged attention matrices.

    Parameters
    ----------
    student_attn : (B, H_s, S, S)
    teacher_attn : (B, H_t, S, S)
    """
    # Average across heads -> (B, S, S)
    s = student_attn.mean(dim=1)
    t = teacher_attn.mean(dim=1)
    return F.mse_loss(s, t)


def cosine_embedding_loss(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
) -> torch.Tensor:
    """1 - cosine similarity, averaged over tokens."""
    s = F.normalize(student_emb, dim=-1)
    t = F.normalize(teacher_emb, dim=-1)
    return (1.0 - (s * t).sum(dim=-1)).mean()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class CombinedDistillationLoss(nn.Module):
    """Weighted combination of all distillation objectives.

    Parameters
    ----------
    alpha : float   Weight for the hard-label MLM cross-entropy loss.
    beta  : float   Weight for the soft-target KL loss.
    gamma : float   Weight for the hidden-state alignment loss.
    delta : float   Weight for the attention-transfer loss.
    temperature : float
    layer_mapping : dict  ``{student_layer_idx: teacher_layer_idx}``
    student_dim, teacher_dim : int
        Hidden dimensions (used to create projection layers).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 0.1,
        delta: float = 0.0,
        temperature: float = 4.0,
        layer_mapping: Optional[Dict[int, int]] = None,
        student_dim: int = 256,
        teacher_dim: int = 512,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.temperature = temperature
        self.layer_mapping = layer_mapping or {}

        # Create per-layer projection when dimensions differ
        self.projections = nn.ModuleDict()
        if student_dim != teacher_dim and self.gamma > 0:
            for s_idx in self.layer_mapping:
                proj = nn.Linear(student_dim, teacher_dim)
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)
                self.projections[str(s_idx)] = proj

    def forward(
        self,
        student_out: Dict[str, object],
        teacher_out: Dict[str, object],
        labels: torch.Tensor,
        masked_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the combined loss.

        Parameters
        ----------
        student_out, teacher_out : dicts from model.forward(...)
        labels : (B, S)  ground-truth token ids (with -100 for ignore).
        masked_positions : (B, S) bool mask of MLM-masked positions.

        Returns
        -------
        total_loss : scalar
        components : dict of individual (unweighted) loss values for logging.
        """
        components: Dict[str, float] = {}

        # ---- 1. Hard-label MLM loss ----
        s_logits = student_out["mlm_logits"]
        task_loss = F.cross_entropy(
            s_logits.view(-1, s_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        components["mlm_loss"] = task_loss.item()

        total = self.alpha * task_loss

        # ---- 2. Soft-target loss ----
        if self.beta > 0:
            t_logits = teacher_out["mlm_logits"]
            # If vocab sizes differ (gene modules), skip soft-target loss
            if s_logits.size(-1) == t_logits.size(-1):
                st_loss = soft_target_loss(
                    s_logits, t_logits, self.temperature, label_mask=masked_positions,
                )
                components["soft_target_loss"] = st_loss.item()
                total = total + self.beta * st_loss

        # ---- 3. Hidden-state loss ----
        if self.gamma > 0 and self.layer_mapping:
            s_hidden = student_out.get("hidden_states", [])
            t_hidden = teacher_out.get("hidden_states", [])
            hs_loss = torch.tensor(0.0, device=total.device)
            n_mapped = 0
            for s_idx, t_idx in self.layer_mapping.items():
                if s_idx < len(s_hidden) and t_idx < len(t_hidden):
                    key = str(s_idx)
                    proj = self.projections[key] if key in self.projections else None
                    hs_loss = hs_loss + hidden_state_loss(
                        s_hidden[s_idx], t_hidden[t_idx], projection=proj,
                    )
                    n_mapped += 1
            if n_mapped > 0:
                hs_loss = hs_loss / n_mapped
                components["hidden_state_loss"] = hs_loss.item()
                total = total + self.gamma * hs_loss

        # ---- 4. Attention-transfer loss ----
        if self.delta > 0 and self.layer_mapping:
            s_attn = student_out.get("attentions", [])
            t_attn = teacher_out.get("attentions", [])
            at_loss = torch.tensor(0.0, device=total.device)
            n_mapped = 0
            for s_idx, t_idx in self.layer_mapping.items():
                if s_idx < len(s_attn) and t_idx < len(t_attn):
                    at_loss = at_loss + attention_transfer_loss(
                        s_attn[s_idx], t_attn[t_idx],
                    )
                    n_mapped += 1
            if n_mapped > 0:
                at_loss = at_loss / n_mapped
                components["attention_loss"] = at_loss.item()
                total = total + self.delta * at_loss

        components["total_loss"] = total.item()
        return total, components
