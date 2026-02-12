"""
Nicheformer Student Model.

A structurally identical but smaller Transformer encoder, designed to be
trained via knowledge distillation from :class:`NicheformerTeacher`.

Default configuration (4 layers, 256-dim, 8 heads) gives roughly a
**4-8x** parameter reduction vs. the teacher.  When combined with
:class:`GeneModuleTokenizer`, the vocabulary shrinks from ~20 k to ~500
tokens, yielding an additional large reduction in embedding-table size.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..config import StudentConfig
from .teacher import SinusoidalPositionalEncoding, TransformerLayerWithAttn

logger = logging.getLogger(__name__)


class NicheformerStudent(nn.Module):
    """Reduced-size Nicheformer for efficient single-cell / spatial inference.

    Shares the same API as :class:`NicheformerTeacher` so that
    the distiller and benchmark can treat them interchangeably.

    Parameters
    ----------
    config : StudentConfig
        Student hyper-parameters.
    """

    def __init__(self, config: Optional[StudentConfig] = None):
        super().__init__()
        if config is None:
            config = StudentConfig()
        self.config = config
        c = config

        vocab_size = c.n_tokens + c.aux_tokens
        if c.use_gene_modules:
            # With gene modules the vocabulary is n_modules + aux_tokens
            vocab_size = c.n_modules + c.aux_tokens

        self.vocab_size = vocab_size

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, c.dim_model, padding_idx=1)

        # Positional encoding
        if c.learnable_pe:
            self.pos_encoder = nn.Embedding(c.context_length, c.dim_model)
        else:
            self.pos_encoder = SinusoidalPositionalEncoding(
                c.dim_model, max_len=c.context_length, dropout=c.dropout,
            )
        self.learnable_pe = c.learnable_pe

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerLayerWithAttn(
                d_model=c.dim_model,
                nhead=c.nheads,
                dim_feedforward=c.dim_feedforward,
                dropout=c.dropout,
            )
            for _ in range(c.nlayers)
        ])

        # Output head — predicts over the full vocabulary (including aux tokens)
        self.classifier = nn.Linear(c.dim_model, vocab_size)

        # Optional projection to teacher dimension (for hidden-state distillation)
        self._teacher_dim: Optional[int] = None
        self.hidden_proj: Optional[nn.Linear] = None

        self._init_weights()

    # ------------------------------------------------------------------ init

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def set_teacher_projection(self, teacher_dim: int) -> None:
        """Create a linear projection from student dim to teacher dim.

        Used during distillation to align hidden-state dimensions.
        """
        self._teacher_dim = teacher_dim
        self.hidden_proj = nn.Linear(self.config.dim_model, teacher_dim)
        nn.init.xavier_uniform_(self.hidden_proj.weight)
        nn.init.zeros_(self.hidden_proj.bias)

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Dict[str, object]:
        """Same signature / return schema as :class:`NicheformerTeacher`."""
        B, S = input_ids.shape

        x = self.embedding(input_ids)
        if self.learnable_pe:
            positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_encoder(positions)
        else:
            x = self.pos_encoder(x)

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

    # ----------------------------------------------------------- convenience

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        layer: int = -1,
        skip_context_tokens: int = 3,
    ) -> torch.Tensor:
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

    def project_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project student hidden state to teacher dimension."""
        if self.hidden_proj is None:
            return hidden
        return self.hidden_proj(hidden)

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str) -> None:
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict(),
        }, path)
        logger.info("Student saved to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "NicheformerStudent":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model
