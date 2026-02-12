"""
Configuration dataclasses for Nicheformer-Lite.

Defines all hyperparameters for the teacher model, student model,
distillation training, and benchmarking.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class NicheformerConfig:
    """Configuration matching the original Nicheformer architecture.

    Default values correspond to the pretrained 110M-cell model from
    Schaar, Tejada-Lapuerta et al. (2024).
    """
    n_tokens: int = 20340
    dim_model: int = 512
    nheads: int = 16
    dim_feedforward: int = 1024
    nlayers: int = 12
    dropout: float = 0.0
    context_length: int = 1500
    learnable_pe: bool = True
    aux_tokens: int = 30
    masking_p: float = 0.15
    use_species_token: bool = True
    use_assay_token: bool = True
    use_modality_token: bool = True


@dataclass
class StudentConfig:
    """Configuration for the reduced student model.

    Defaults give a ~4x parameter reduction compared to the teacher.
    When ``use_gene_modules=True``, vocabulary compression yields an
    additional ~10-40x embedding-table reduction.
    """
    n_tokens: int = 20340       # overridden when using gene modules
    dim_model: int = 256
    nheads: int = 8
    dim_feedforward: int = 512
    nlayers: int = 4
    dropout: float = 0.1
    context_length: int = 1500
    learnable_pe: bool = True
    aux_tokens: int = 30
    masking_p: float = 0.15
    use_gene_modules: bool = False
    n_modules: int = 500


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation training."""
    # Loss weights
    temperature: float = 4.0
    alpha: float = 0.5          # hard-label MLM loss
    beta: float = 0.5           # soft-target KL loss
    gamma: float = 0.1          # hidden-state alignment loss
    delta: float = 0.0          # attention-transfer loss
    # Optimiser
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    # Training
    batch_size: int = 32
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    # Layer mapping  student_layer_idx -> teacher_layer_idx
    layer_mapping: Optional[Dict[int, int]] = None
    # Misc
    checkpoint_dir: str = "./distillation_checkpoints"
    log_every: int = 100
    save_every: int = 5000
    eval_every: int = 1000
    device: str = "cpu"
    seed: int = 42


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmarking suite."""
    n_warmup_runs: int = 10
    n_benchmark_runs: int = 100
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32])
    seq_lengths: List[int] = field(default_factory=lambda: [256, 512, 1024])
    device: str = "cpu"
    output_dir: str = "./benchmark_results"
    n_synthetic_cells: int = 2000
    n_genes: int = 20340
    save_plots: bool = True
