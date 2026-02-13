# Nicheformer-Lite

A toolkit for compressing the [Nicheformer](https://github.com/theislab/nicheformer) foundation model (Schaar, Tejada-Lapuerta et al., 2024) through **knowledge distillation** and **gene-module tokenization**.

The resulting student model is **3.6x smaller** and **4.6x faster** while retaining strong single-cell / spatial-omics representation quality.

| Metric | Teacher | Student | Improvement |
|---|---|---|---|
| Parameters | 47.1 M | 13.1 M | 3.6x smaller |
| Memory (float32) | 188.6 MB | 52.3 MB | 3.6x less |
| Latency (batch 32, seq 512) | 995 ms | 215 ms | 4.6x faster |
| CKA similarity | - | - | 0.85 |

## Installation

```bash
git clone https://github.com/echang992000/nicheformer_lite.git
cd nicheformer_lite
pip install -e .
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- NumPy, SciPy, scikit-learn, matplotlib, tqdm

## Quick Start

### 1. Distill a smaller model

```python
from nicheformer_lite import (
    NicheformerTeacher, NicheformerStudent,
    NicheformerDistiller, NicheformerConfig, StudentConfig,
)

# Load or create teacher (use from_pretrained with a real checkpoint)
teacher = NicheformerTeacher()  # or NicheformerTeacher.from_pretrained("nicheformer.ckpt")

# Create student (4 layers, 256-dim vs teacher's 12 layers, 512-dim)
student = NicheformerStudent()

# Run distillation
distiller = NicheformerDistiller(teacher, student)
distiller.train(dataloader, max_steps=10_000)
```

### 2. Gene-module tokenization

```python
from nicheformer_lite import GeneModuleTokenizer

# Group ~20k genes into 500 functional modules
tokenizer = GeneModuleTokenizer(n_modules=500, strategy="correlation")
tokenizer.fit(expression_matrix, gene_names=gene_list)

# Tokenize at the module level (shorter sequences, smaller vocabulary)
module_tokens = tokenizer.transform(expression_matrix)

# Decode back to gene level
gene_predictions = tokenizer.decode_to_genes(module_values)
```

Supported grouping strategies:

| Strategy | Method | Best for |
|---|---|---|
| `correlation` | Hierarchical clustering on gene-gene Pearson correlation | Capturing co-expression modules |
| `pca` | PCA loadings define soft module memberships | Variance-driven grouping |
| `kmeans` | K-means on gene expression vectors | Fast, balanced clusters |
| `random` | Uniform-random assignment | Baseline / ablation |

A **learnable** variant (`LearnableGeneModuleLayer`) uses Gumbel-Softmax for differentiable gene-to-module assignment, trainable end-to-end with the student model.

### 3. Benchmark teacher vs student

```python
from nicheformer_lite import NicheformerBenchmark

benchmark = NicheformerBenchmark(teacher, student)
report = benchmark.run_all()
print(report.summary())
benchmark.plot_results("./results")
```

## CLI Benchmark Script

```bash
# Quick benchmark with synthetic data
python scripts/benchmark_nicheformer.py

# Distillation + benchmark + plots
python scripts/benchmark_nicheformer.py --distill --distill-steps 5000 --save-plots

# With gene-module tokenization
python scripts/benchmark_nicheformer.py --gene-modules --n-modules 500 --module-strategy correlation

# Load real pretrained Nicheformer weights
python scripts/benchmark_nicheformer.py --teacher-checkpoint path/to/nicheformer.ckpt --distill

# Custom student architecture
python scripts/benchmark_nicheformer.py --student-layers 6 --student-dim 384 --student-heads 12

# GPU acceleration
python scripts/benchmark_nicheformer.py --device cuda
```

## Architecture

### Knowledge Distillation

The student learns from the teacher through three complementary loss signals:

- **Soft-target loss** (Hinton et al., 2015) -- KL divergence on temperature-scaled logits transfers the teacher's "dark knowledge"
- **Hidden-state loss** -- MSE between aligned intermediate representations with learned projection layers
- **Attention transfer** (Zagoruyko & Komodakis, 2017) -- MSE on head-averaged attention maps

```
Loss = alpha * MLM_CE + beta * KL_soft + gamma * MSE_hidden + delta * MSE_attn
```

### Model Configurations

| | Teacher (default) | Student (default) |
|---|---|---|
| Layers | 12 | 4 |
| Hidden dim | 512 | 256 |
| Attention heads | 16 | 8 |
| FFN dim | 1024 | 512 |
| Vocabulary | 20,370 | 20,370 (or ~530 with gene modules) |

## Package Structure

```
nicheformer_lite/
├── nicheformer_lite/           # Python package
│   ├── __init__.py             # Public API
│   ├── config.py               # Configuration dataclasses
│   ├── models/
│   │   ├── tokenizer.py        # GeneModuleTokenizer + LearnableGeneModuleLayer
│   │   ├── teacher.py          # NicheformerTeacher (full-size)
│   │   └── student.py          # NicheformerStudent (reduced)
│   ├── distillation/
│   │   ├── losses.py           # Soft-target, hidden-state, attention losses
│   │   └── distiller.py        # NicheformerDistiller training loop
│   └── benchmark/
│       └── evaluator.py        # NicheformerBenchmark suite
├── scripts/
│   └── benchmark_nicheformer.py
├── setup.py
├── requirements.txt
└── README.md
```

## Benchmark Output

The benchmark script produces:

- `benchmark_report.json` -- Full metrics in JSON
- `size_comparison.png` -- Parameter and memory bar charts
- `speed_comparison.png` -- Latency comparison across batch sizes
- `quality_comparison.png` -- MLM accuracy, top-5 recall, CKA similarity

## Citation

This package builds upon the Nicheformer foundation model:

```bibtex
@article{schaar2024nicheformer,
  title={Nicheformer: a foundation model for single-cell and spatial omics},
  author={Schaar, A.C. and Tejada-Lapuerta, A. and others},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.04.15.589472}
}
```

## License

MIT
