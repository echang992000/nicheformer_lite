"""
Configuration dataclasses for Nicheformer-Lite.

Defines all hyperparameters for the teacher model, student model,
distillation training, and benchmarking.  Each config validates its
values in ``__post_init__`` so that errors surface early.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


def _check_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _check_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _check_divisible(name_a: str, a: int, name_b: str, b: int) -> None:
    if a % b != 0:
        raise ValueError(f"{name_a} ({a}) must be divisible by {name_b} ({b})")


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

    def __post_init__(self) -> None:
        _check_positive("n_tokens", self.n_tokens)
        _check_positive("dim_model", self.dim_model)
        _check_positive("nheads", self.nheads)
        _check_positive("nlayers", self.nlayers)
        _check_divisible("dim_model", self.dim_model, "nheads", self.nheads)
        _check_non_negative("dropout", self.dropout)
        if not 0.0 <= self.masking_p <= 1.0:
            raise ValueError(f"masking_p must be in [0, 1], got {self.masking_p}")


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

    def __post_init__(self) -> None:
        _check_positive("dim_model", self.dim_model)
        _check_positive("nheads", self.nheads)
        _check_positive("nlayers", self.nlayers)
        _check_divisible("dim_model", self.dim_model, "nheads", self.nheads)
        _check_non_negative("dropout", self.dropout)
        if self.use_gene_modules:
            _check_positive("n_modules", self.n_modules)


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
    # Early stopping
    early_stopping: bool = False
    patience: int = 10          # eval rounds without improvement before stop
    # Mixed precision
    use_amp: bool = False
    # Layer mapping  student_layer_idx -> teacher_layer_idx
    layer_mapping: Optional[Dict[int, int]] = None
    # Misc
    checkpoint_dir: str = "./distillation_checkpoints"
    log_every: int = 100
    save_every: int = 5000
    eval_every: int = 1000
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self) -> None:
        _check_positive("temperature", self.temperature)
        _check_non_negative("alpha", self.alpha)
        _check_non_negative("beta", self.beta)
        _check_non_negative("gamma", self.gamma)
        _check_non_negative("delta", self.delta)
        _check_positive("learning_rate", self.learning_rate)
        _check_positive("max_steps", self.max_steps)
        _check_positive("batch_size", self.batch_size)
        _check_positive("gradient_accumulation", self.gradient_accumulation)
        if self.device not in ("cpu", "cuda", "mps"):
            raise ValueError(f"device must be cpu/cuda/mps, got '{self.device}'")


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

    def __post_init__(self) -> None:
        _check_positive("n_warmup_runs", self.n_warmup_runs)
        _check_positive("n_benchmark_runs", self.n_benchmark_runs)
        if not self.batch_sizes:
            raise ValueError("batch_sizes must not be empty")
