"""
Nicheformer-Lite: Reduced-size Nicheformer via distillation & gene-module tokenisation.

A toolkit for compressing the Nicheformer foundation model (Schaar et al., 2024)
through knowledge distillation and gene-module tokenisation.  The resulting
student model is 4-40x smaller and faster while retaining strong single-cell /
spatial-omics representation quality.

Quick start
-----------
>>> from nicheformer_lite import (
...     NicheformerTeacher, NicheformerStudent, GeneModuleTokenizer,
...     NicheformerDistiller, NicheformerBenchmark,
... )
>>>
>>> teacher = NicheformerTeacher()                   # full model
>>> student = NicheformerStudent()                   # 4x smaller
>>> distiller = NicheformerDistiller(teacher, student)
>>> distiller.train(dataloader, max_steps=10_000)
>>>
>>> benchmark = NicheformerBenchmark(teacher, student)
>>> report = benchmark.run_all()
>>> print(report.summary())
"""

__version__ = "0.1.0"

from .config import (
    BenchmarkConfig,
    DistillationConfig,
    NicheformerConfig,
    StudentConfig,
)
from .models import GeneModuleTokenizer, NicheformerStudent, NicheformerTeacher
from .models.tokenizer import LearnableGeneModuleLayer
from .distillation import NicheformerDistiller, CombinedDistillationLoss
from .benchmark import NicheformerBenchmark

__all__ = [
    # Config
    "NicheformerConfig",
    "StudentConfig",
    "DistillationConfig",
    "BenchmarkConfig",
    # Models
    "NicheformerTeacher",
    "NicheformerStudent",
    "GeneModuleTokenizer",
    "LearnableGeneModuleLayer",
    # Distillation
    "NicheformerDistiller",
    "CombinedDistillationLoss",
    # Benchmark
    "NicheformerBenchmark",
]
