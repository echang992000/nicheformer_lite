#!/usr/bin/env python3
"""
Benchmark script: Nicheformer (original) vs Nicheformer-Lite (reduced).

This script runs an end-to-end comparison of the full Nicheformer model against
a distilled student variant, measuring model size, inference speed, and
prediction quality.

Usage
-----
# Quick benchmark with defaults (synthetic data, CPU)
python scripts/benchmark_nicheformer.py

# Full benchmark with distillation training
python scripts/benchmark_nicheformer.py --distill --distill-steps 5000

# Use a pretrained teacher checkpoint
python scripts/benchmark_nicheformer.py --teacher-checkpoint path/to/nicheformer.ckpt

# Gene-module tokenisation (500 modules via co-expression clustering)
python scripts/benchmark_nicheformer.py --gene-modules --n-modules 500 --module-strategy correlation

# Custom student architecture
python scripts/benchmark_nicheformer.py --student-layers 6 --student-dim 384 --student-heads 12

# Benchmark on GPU
python scripts/benchmark_nicheformer.py --device cuda

# Save results
python scripts/benchmark_nicheformer.py --output-dir ./results --save-plots

Run from the nicheformer_lite/ root directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from nicheformer_lite import (
    BenchmarkConfig,
    DistillationConfig,
    NicheformerBenchmark,
    NicheformerConfig,
    NicheformerDistiller,
    NicheformerStudent,
    NicheformerTeacher,
    GeneModuleTokenizer,
    StudentConfig,
)
from nicheformer_lite.distillation.distiller import (
    generate_synthetic_data,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark Nicheformer vs Nicheformer-Lite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Teacher
    g = p.add_argument_group("Teacher model")
    g.add_argument(
        "--teacher-checkpoint", type=str, default=None,
        help="Path to pretrained Nicheformer .ckpt (optional).",
    )
    g.add_argument("--teacher-layers", type=int, default=12)
    g.add_argument("--teacher-dim", type=int, default=512)
    g.add_argument("--teacher-heads", type=int, default=16)
    g.add_argument("--teacher-ffn", type=int, default=1024)
    g.add_argument("--n-tokens", type=int, default=20340)

    # Student
    g = p.add_argument_group("Student model")
    g.add_argument("--student-layers", type=int, default=4)
    g.add_argument("--student-dim", type=int, default=256)
    g.add_argument("--student-heads", type=int, default=8)
    g.add_argument("--student-ffn", type=int, default=512)
    g.add_argument("--student-dropout", type=float, default=0.1)

    # Gene module tokenisation
    g = p.add_argument_group("Gene module tokenisation")
    g.add_argument(
        "--gene-modules", action="store_true",
        help="Enable gene-module tokenisation for the student.",
    )
    g.add_argument("--n-modules", type=int, default=500)
    g.add_argument(
        "--module-strategy", type=str, default="kmeans",
        choices=["correlation", "pca", "kmeans", "random"],
    )

    # Distillation
    g = p.add_argument_group("Distillation")
    g.add_argument(
        "--distill", action="store_true",
        help="Run distillation training before benchmarking.",
    )
    g.add_argument("--distill-steps", type=int, default=2000)
    g.add_argument("--temperature", type=float, default=4.0)
    g.add_argument("--alpha", type=float, default=0.5, help="Hard-label loss weight.")
    g.add_argument("--beta", type=float, default=0.5, help="Soft-target loss weight.")
    g.add_argument("--gamma", type=float, default=0.1, help="Hidden-state loss weight.")
    g.add_argument("--delta", type=float, default=0.0, help="Attention loss weight.")
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--batch-size", type=int, default=32)

    # Benchmark
    g = p.add_argument_group("Benchmark settings")
    g.add_argument("--device", type=str, default="cpu",
                   choices=["cpu", "cuda", "mps"])
    g.add_argument("--n-cells", type=int, default=2000,
                   help="Number of synthetic cells for quality evaluation.")
    g.add_argument("--seq-len", type=int, default=512)
    g.add_argument("--warmup-runs", type=int, default=5)
    g.add_argument("--bench-runs", type=int, default=50)
    g.add_argument("--output-dir", type=str, default="./benchmark_results")
    g.add_argument("--save-plots", action="store_true")
    g.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resolve device ----
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device = "cpu"
    if device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        logger.warning("MPS not available, falling back to CPU.")
        device = "cpu"

    logger.info("Device: %s", device)

    # ==================================================================
    # 1. Build teacher
    # ==================================================================
    teacher_config = NicheformerConfig(
        n_tokens=args.n_tokens,
        dim_model=args.teacher_dim,
        nheads=args.teacher_heads,
        dim_feedforward=args.teacher_ffn,
        nlayers=args.teacher_layers,
    )

    if args.teacher_checkpoint:
        logger.info("Loading teacher from %s", args.teacher_checkpoint)
        teacher = NicheformerTeacher.from_pretrained(
            args.teacher_checkpoint, config=teacher_config,
        )
    else:
        logger.info("Creating randomly-initialised teacher (no checkpoint).")
        teacher = NicheformerTeacher(teacher_config)

    t_params = teacher.count_parameters(trainable_only=False)
    logger.info("Teacher: %d params (%.1f M)", t_params, t_params / 1e6)

    # ==================================================================
    # 2. Build student
    # ==================================================================
    student_config = StudentConfig(
        n_tokens=args.n_tokens,
        dim_model=args.student_dim,
        nheads=args.student_heads,
        dim_feedforward=args.student_ffn,
        nlayers=args.student_layers,
        dropout=args.student_dropout,
        use_gene_modules=args.gene_modules,
        n_modules=args.n_modules,
    )

    student = NicheformerStudent(student_config)
    s_params = student.count_parameters(trainable_only=False)
    logger.info("Student: %d params (%.1f M)", s_params, s_params / 1e6)
    logger.info(
        "Compression ratio: %.1fx", t_params / max(s_params, 1),
    )

    # ==================================================================
    # 3. Gene module tokenisation (optional)
    # ==================================================================
    tokenizer = None
    if args.gene_modules:
        logger.info(
            "Fitting gene-module tokenizer (%d modules, strategy=%s)...",
            args.n_modules, args.module_strategy,
        )
        # Generate synthetic expression matrix for fitting
        rng = np.random.RandomState(args.seed)
        n_fit_cells = min(args.n_cells, 5000)
        synthetic_expr = rng.exponential(1.0, size=(n_fit_cells, args.n_tokens))
        # Add some correlation structure (block-diagonal)
        block_size = args.n_tokens // args.n_modules
        for b in range(args.n_modules):
            start = b * block_size
            end = min(start + block_size, args.n_tokens)
            factor = rng.randn(n_fit_cells, 1) * 2
            synthetic_expr[:, start:end] += np.abs(factor)

        tokenizer = GeneModuleTokenizer(
            n_modules=args.n_modules,
            strategy=args.module_strategy,
            random_state=args.seed,
        )
        tokenizer.fit(synthetic_expr)

        sizes = tokenizer.module_sizes()
        logger.info(
            "Module sizes: min=%d, max=%d, median=%d",
            sizes.min(), sizes.max(), int(np.median(sizes)),
        )

    # ==================================================================
    # 4. Distillation (optional)
    # ==================================================================
    if args.distill:
        logger.info("=" * 60)
        logger.info("  KNOWLEDGE DISTILLATION")
        logger.info("=" * 60)

        distill_config = DistillationConfig(
            temperature=args.temperature,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            delta=args.delta,
            learning_rate=args.lr,
            max_steps=args.distill_steps,
            batch_size=args.batch_size,
            device=device,
            checkpoint_dir=str(output_dir / "distillation_checkpoints"),
            log_every=max(1, args.distill_steps // 20),
            save_every=max(1, args.distill_steps // 5),
            eval_every=max(1, args.distill_steps // 10),
            seed=args.seed,
        )

        distiller = NicheformerDistiller(
            teacher=teacher,
            student=student,
            config=distill_config,
            tokenizer=tokenizer,
        )

        # Generate training data
        logger.info("Generating %d synthetic training cells...", args.n_cells)
        train_tokens = generate_synthetic_data(
            n_cells=args.n_cells,
            n_tokens=args.n_tokens,
            seq_len=args.seq_len,
        )
        eval_tokens = generate_synthetic_data(
            n_cells=min(500, args.n_cells // 4),
            n_tokens=args.n_tokens,
            seq_len=args.seq_len,
        )

        train_loader = NicheformerDistiller.make_dataloader(
            train_tokens, batch_size=args.batch_size,
        )
        eval_loader = NicheformerDistiller.make_dataloader(
            eval_tokens, batch_size=args.batch_size, shuffle=False,
        )

        t0 = time.time()
        result = distiller.train(
            train_loader,
            max_steps=args.distill_steps,
            eval_dataloader=eval_loader,
        )
        elapsed = time.time() - t0
        logger.info("Distillation completed in %.1f s", elapsed)

        # The student is now updated in place
        student = distiller.student

    # ==================================================================
    # 5. Benchmark
    # ==================================================================
    logger.info("=" * 60)
    logger.info("  BENCHMARKING")
    logger.info("=" * 60)

    bench_config = BenchmarkConfig(
        n_warmup_runs=args.warmup_runs,
        n_benchmark_runs=args.bench_runs,
        batch_sizes=[1, 8, 32],
        seq_lengths=[256, 512],
        device=device,
        output_dir=str(output_dir),
        n_synthetic_cells=args.n_cells,
        save_plots=args.save_plots,
    )

    benchmark = NicheformerBenchmark(
        teacher=teacher,
        student=student,
        config=bench_config,
    )

    # Size
    logger.info("Measuring model size...")
    benchmark.benchmark_size()

    # Speed
    logger.info("Measuring inference speed...")
    benchmark.benchmark_speed()

    # Quality
    logger.info("Measuring prediction quality...")
    quality_data = generate_synthetic_data(
        n_cells=args.n_cells,
        n_tokens=args.n_tokens,
        seq_len=args.seq_len,
    )
    benchmark.benchmark_quality(quality_data)

    # Report
    report = benchmark.report
    print(report.summary())

    # Save
    report.save(output_dir / "benchmark_report.json")
    logger.info("Report saved to %s", output_dir / "benchmark_report.json")

    if args.save_plots:
        logger.info("Generating plots...")
        benchmark.plot_results(output_dir, show=False)

    # ==================================================================
    # 6. Gene-module analysis (if applicable)
    # ==================================================================
    if tokenizer is not None:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  GENE MODULE TOKENISATION ANALYSIS")
        logger.info("=" * 60)

        sizes = tokenizer.module_sizes()
        logger.info("Modules: %d", tokenizer.n_modules)
        logger.info("Genes: %d", tokenizer.n_genes)
        logger.info(
            "Module sizes: min=%d, max=%d, mean=%.1f, median=%d",
            sizes.min(), sizes.max(), sizes.mean(), int(np.median(sizes)),
        )
        logger.info(
            "Vocabulary reduction: %d -> %d tokens (%.1fx)",
            args.n_tokens + 30,
            tokenizer.vocab_size,
            (args.n_tokens + 30) / tokenizer.vocab_size,
        )

        # Save tokenizer
        tok_path = output_dir / "gene_module_tokenizer.npz"
        tokenizer.save(tok_path)
        logger.info("Tokenizer saved to %s", tok_path)

    logger.info("")
    logger.info("All results saved to %s", output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
