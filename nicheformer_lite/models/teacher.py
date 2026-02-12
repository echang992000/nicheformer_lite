"""
Nicheformer Teacher Model.

A faithful re-implementation of the Nicheformer encoder architecture
(Schaar, Tejada-Lapuerta et al., 2024) that can:

  * Be instantiated from scratch for testing / ablations.
  * Load official pretrained weights from a Lightning checkpoint.
  * Expose intermediate hidden states and attention maps for distillation.

The architecture mirrors ``src/nicheformer/models/_nicheformer.py`` in the
official repository:  Embedding  ->  Positional Encoding  ->
TransformerEncoder (12 layers)  ->  Classifier head (MLM).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import NicheformerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Positional encoding (sinusoidal fallback)
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, dim_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dim_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Custom encoder layer that can return attention weights
# ---------------------------------------------------------------------------

class TransformerLayerWithAttn(nn.Module):
    """Pre-norm Transformer encoder layer that optionally returns attention
    weights, which ``nn.TransformerEncoderLayer`` does not expose easily."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        need_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-norm self-attention
        x = self.norm1(src)
        x2, attn_weights = self.self_attn(
            x, x, x,
            key_padding_mask=src_key_padding_mask,
            need_weights=need_attn_weights,
            average_attn_weights=False,  # return per-head weights
        )
        src = src + self.dropout(x2)

        # Pre-norm FFN
        x = self.norm2(src)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        src = src + self.dropout(x)

        return src, attn_weights


# ---------------------------------------------------------------------------
# Teacher model
# ---------------------------------------------------------------------------

class NicheformerTeacher(nn.Module):
    """Full-size Nicheformer encoder for use as a distillation teacher.

    Parameters
    ----------
    config : NicheformerConfig
        Model hyper-parameters (defaults match the published 110 M-cell model).
    """

    def __init__(self, config: Optional[NicheformerConfig] = None):
        super().__init__()
        if config is None:
            config = NicheformerConfig()
        self.config = config
        c = config

        # Token embedding (gene tokens + aux special tokens)
        self.embedding = nn.Embedding(
            c.n_tokens + c.aux_tokens, c.dim_model, padding_idx=1,
        )
        # Positional encoding
        if c.learnable_pe:
            self.pos_encoder = nn.Embedding(c.context_length, c.dim_model)
        else:
            self.pos_encoder = SinusoidalPositionalEncoding(
                c.dim_model, max_len=c.context_length, dropout=c.dropout,
            )
        self.learnable_pe = c.learnable_pe

        # Transformer encoder
        self.layers = nn.ModuleList([
            TransformerLayerWithAttn(
                d_model=c.dim_model,
                nhead=c.nheads,
                dim_feedforward=c.dim_feedforward,
                dropout=c.dropout,
            )
            for _ in range(c.nlayers)
        ])

        # MLM classifier head (covers full vocabulary including aux tokens)
        self.classifier = nn.Linear(c.dim_model, c.n_tokens + c.aux_tokens)

        # Pooler for downstream tasks
        self.pooler_head = nn.Sequential(
            nn.Linear(c.dim_model, c.dim_model),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ----- forward ----------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        input_ids : (batch, seq_len) int64 token indices.
        padding_mask : (batch, seq_len) bool — True for PAD positions.
        output_hidden_states : return list of per-layer hidden states.
        output_attentions : return list of per-layer attention weights.

        Returns
        -------
        dict with keys:
            ``mlm_logits``       (batch, seq_len, n_tokens)
            ``last_hidden``      (batch, seq_len, dim_model)
            ``hidden_states``    list of (batch, seq_len, dim_model)  [optional]
            ``attentions``       list of (batch, nheads, seq_len, seq_len) [optional]
        """
        B, S = input_ids.shape

        # --- embeddings ---
        x = self.embedding(input_ids)
        if self.learnable_pe:
            positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_encoder(positions)
        else:
            x = self.pos_encoder(x)

        # --- encoder ---
        hidden_states: List[torch.Tensor] = []
        attentions: List[torch.Tensor] = []

        if output_hidden_states:
            hidden_states.append(x)

        for layer in self.layers:
            x, attn_w = layer(
                x,
                src_key_padding_mask=padding_mask,
                need_attn_weights=output_attentions,
            )
            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions and attn_w is not None:
                attentions.append(attn_w)

        # --- heads ---
        mlm_logits = self.classifier(x)

        out: Dict[str, object] = {
            "mlm_logits": mlm_logits,
            "last_hidden": x,
        }
        if output_hidden_states:
            out["hidden_states"] = hidden_states
        if output_attentions:
            out["attentions"] = attentions
        return out

    # ----- convenience -------------------------------------------------------

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        layer: int = -1,
        skip_context_tokens: int = 3,
    ) -> torch.Tensor:
        """Mean-pooled embeddings from a specific layer.

        Parameters
        ----------
        layer : int
            Layer index (0 = after embedding, -1 = last encoder layer).
        skip_context_tokens : int
            Number of leading context tokens (species, assay, modality)
            to exclude from pooling.
        """
        out = self.forward(input_ids, padding_mask, output_hidden_states=True)
        hs = out["hidden_states"][layer]
        if skip_context_tokens > 0:
            hs = hs[:, skip_context_tokens:]
            if padding_mask is not None:
                padding_mask = padding_mask[:, skip_context_tokens:]

        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(-1).float()
            return (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return hs.mean(dim=1)

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    # ----- checkpoint loading ------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config: Optional[NicheformerConfig] = None,
        strict: bool = False,
    ) -> "NicheformerTeacher":
        """Load from an official Nicheformer Lightning checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to a ``.ckpt`` file produced by ``pytorch_lightning``.
        config : NicheformerConfig, optional
            Override default config.  If ``None``, inferred from the
            checkpoint hyper-parameters when available.
        strict : bool
            Whether to require an exact state-dict match.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Try to infer config from checkpoint hparams
        if config is None:
            hp = ckpt.get("hyper_parameters", {})
            config = NicheformerConfig(
                n_tokens=hp.get("n_tokens", 20340),
                dim_model=hp.get("dim_model", 512),
                nheads=hp.get("nheads", 16),
                dim_feedforward=hp.get("dim_feedforward", 1024),
                nlayers=hp.get("nlayers", 12),
                dropout=hp.get("dropout", 0.0),
                context_length=hp.get("context_length", 1500),
                learnable_pe=hp.get("learnable_pe", True),
                aux_tokens=hp.get("aux_tokens", 30),
            )

        model = cls(config)

        # Handle Lightning checkpoint state-dict keys
        state_dict = ckpt.get("state_dict", ckpt)
        # Strip "model." prefix that Lightning sometimes adds
        cleaned = {}
        for k, v in state_dict.items():
            new_key = k
            # Lightning wraps model under the LightningModule attribute names
            for prefix in ("model.", "nicheformer.", "net."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break
            cleaned[new_key] = v

        # The original Nicheformer uses nn.TransformerEncoder under "model"
        # key, whereas we use self.layers.  Try to remap.
        remapped = _remap_state_dict(cleaned, config.nlayers)

        missing, unexpected = model.load_state_dict(remapped, strict=strict)
        if missing:
            logger.warning("Missing keys when loading checkpoint: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys in checkpoint: %s", unexpected)

        logger.info(
            "Loaded teacher from %s (%d params)",
            checkpoint_path, model.count_parameters(trainable_only=False),
        )
        return model


def _remap_state_dict(sd: dict, nlayers: int) -> dict:
    """Remap state-dict keys between the official Nicheformer layout
    (``nn.TransformerEncoder``) and our ``nn.ModuleList`` of custom layers."""
    out = {}
    for k, v in sd.items():
        new_key = k
        # model.layers.N.* -> layers.N.*
        if new_key.startswith("model.layers."):
            new_key = new_key[len("model."):]
        # encoder.layers.N.* -> layers.N.*
        if new_key.startswith("encoder.layers."):
            new_key = new_key[len("encoder."):]
        out[new_key] = v
    return out
