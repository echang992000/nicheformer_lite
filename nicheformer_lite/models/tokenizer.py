"""
Gene Module Tokenizer for Nicheformer-Lite.

Instead of treating each gene as an individual token (vocabulary ~20 k),
this module groups genes into functional modules and tokenises at the
module level.  This yields:

  * Smaller vocabulary  (e.g. 500 modules vs 20 340 genes)
  * Shorter sequences   (at most n_modules tokens per cell)
  * Faster attention     (quadratic in sequence length)

Four grouping strategies are provided:

  ``correlation``  Hierarchical clustering on gene-gene Pearson correlation.
  ``pca``          PCA loadings define soft module memberships; genes are
                   assigned to the component with highest absolute loading.
  ``kmeans``       K-means on the gene-expression vectors (genes as samples).
  ``random``       Uniform-random assignment (baseline / ablation).

After fitting, :meth:`transform` converts a cells-by-genes matrix into
rank-based module tokens, following the same rank-tokenisation scheme
as the original Nicheformer (Schaar et al., 2024).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dense(X: np.ndarray) -> np.ndarray:
    """Ensure *X* is a dense float32 array."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _subsample_rows(X: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
    if X.shape[0] <= n:
        return _to_dense(X)
    idx = rng.choice(X.shape[0], n, replace=False)
    return _to_dense(X[idx])


# ---------------------------------------------------------------------------
# Core tokeniser
# ---------------------------------------------------------------------------

class GeneModuleTokenizer:
    """Groups genes into functional modules for compressed tokenisation.

    Parameters
    ----------
    n_modules : int
        Number of gene modules to create.
    strategy : str
        One of ``'correlation'``, ``'pca'``, ``'kmeans'``, ``'random'``.
    max_fit_cells : int
        Maximum number of cells used when fitting (for speed).
    random_state : int
        Seed for reproducibility.
    aux_tokens : int
        Number of reserved special-token indices (PAD, MASK, CLS, ...).
        Nicheformer uses 30 by default.
    """

    def __init__(
        self,
        n_modules: int = 500,
        strategy: str = "correlation",
        max_fit_cells: int = 5000,
        random_state: int = 42,
        aux_tokens: int = 30,
    ) -> None:
        self.n_modules = n_modules
        self.strategy = strategy
        self.max_fit_cells = max_fit_cells
        self.random_state = random_state
        self.aux_tokens = aux_tokens

        # Populated by fit()
        self.gene_to_module: Optional[np.ndarray] = None  # (n_genes,)
        self.module_weights: Optional[np.ndarray] = None   # (n_modules, n_genes)
        self.gene_names: Optional[np.ndarray] = None
        self.n_genes: Optional[int] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def vocab_size(self) -> int:
        """Total token vocabulary (modules + aux tokens)."""
        return self.n_modules + self.aux_tokens

    def fit(
        self,
        expression_matrix: np.ndarray,
        gene_names: Optional[Sequence[str]] = None,
    ) -> "GeneModuleTokenizer":
        """Learn gene-module assignments from an expression matrix.

        Parameters
        ----------
        expression_matrix : array-like, shape (n_cells, n_genes)
            Raw or normalised expression counts.
        gene_names : sequence of str, optional
            Gene identifiers (for interpretability).

        Raises
        ------
        ValueError
            If the input has fewer than 2 dimensions, contains NaN/Inf,
            or has fewer genes than modules.
        """
        if hasattr(expression_matrix, "ndim"):
            if expression_matrix.ndim != 2:
                raise ValueError(
                    f"expression_matrix must be 2-D (cells, genes), "
                    f"got shape {expression_matrix.shape}"
                )
        X_check = _to_dense(expression_matrix)
        if not np.all(np.isfinite(X_check[:min(1000, len(X_check))])):
            raise ValueError("expression_matrix contains NaN or Inf values")
        if X_check.shape[1] < self.n_modules:
            raise ValueError(
                f"Number of genes ({X_check.shape[1]}) must be >= n_modules "
                f"({self.n_modules})"
            )

        self.n_genes = expression_matrix.shape[1]
        if gene_names is not None:
            self.gene_names = np.asarray(gene_names)
            if len(self.gene_names) != self.n_genes:
                raise ValueError(
                    f"gene_names length ({len(self.gene_names)}) != "
                    f"n_genes ({self.n_genes})"
                )

        rng = np.random.RandomState(self.random_state)
        X_sub = _subsample_rows(expression_matrix, self.max_fit_cells, rng)

        if self.strategy == "correlation":
            self._fit_correlation(X_sub, rng)
        elif self.strategy == "pca":
            self._fit_pca(X_sub, rng)
        elif self.strategy == "kmeans":
            self._fit_kmeans(X_sub, rng)
        elif self.strategy == "random":
            self._fit_random(rng)
        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'")

        self._fitted = True
        logger.info(
            "Fitted %d gene modules (strategy=%s) from %d genes",
            self.n_modules, self.strategy, self.n_genes,
        )
        return self

    def transform(
        self,
        expression_matrix: np.ndarray,
        max_seq_len: Optional[int] = None,
    ) -> np.ndarray:
        """Tokenise expression data at the module level.

        Follows the Nicheformer rank-tokenisation protocol:
        1. Aggregate gene expression per module.
        2. Rank modules by descending aggregated expression.
        3. Return module indices (offset by ``aux_tokens``) as tokens.

        Parameters
        ----------
        expression_matrix : array-like, shape (n_cells, n_genes)
        max_seq_len : int, optional
            Maximum token-sequence length.  Defaults to ``n_modules``.

        Returns
        -------
        tokens : ndarray, shape (n_cells, max_seq_len), dtype int64
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        if max_seq_len is None:
            max_seq_len = self.n_modules

        X = _to_dense(expression_matrix)
        n_cells = X.shape[0]

        # --- aggregate to module level ---
        module_expr = np.zeros((n_cells, self.n_modules), dtype=np.float32)
        for m in range(self.n_modules):
            mask = self.gene_to_module == m
            if mask.any():
                module_expr[:, m] = X[:, mask].mean(axis=1)

        # --- rank-based tokenisation ---
        tokens = np.zeros((n_cells, max_seq_len), dtype=np.int64)
        for i in range(n_cells):
            expr = module_expr[i]
            nz = np.nonzero(expr)[0]
            if len(nz) == 0:
                continue
            ranked = nz[np.argsort(-expr[nz])][:max_seq_len]
            tokens[i, : len(ranked)] = ranked + self.aux_tokens

        return tokens

    def fit_transform(
        self,
        expression_matrix: np.ndarray,
        gene_names: Optional[Sequence[str]] = None,
        max_seq_len: Optional[int] = None,
    ) -> np.ndarray:
        self.fit(expression_matrix, gene_names)
        return self.transform(expression_matrix, max_seq_len)

    def decode_to_genes(
        self,
        module_values: np.ndarray,
    ) -> np.ndarray:
        """Project module-level values back to gene-level.

        Parameters
        ----------
        module_values : array, shape (n_cells, n_modules)

        Returns
        -------
        gene_values : array, shape (n_cells, n_genes)
        """
        if self.module_weights is not None:
            weights = np.abs(self.module_weights)  # (n_modules, n_genes)
            return module_values @ weights
        # fallback: broadcast module value to member genes
        out = np.zeros((module_values.shape[0], self.n_genes), dtype=np.float32)
        for m in range(self.n_modules):
            mask = self.gene_to_module == m
            out[:, mask] = module_values[:, m : m + 1]
        return out

    def get_module_genes(self, module_idx: int) -> np.ndarray:
        """Return gene indices (and optionally names) in a given module."""
        mask = self.gene_to_module == module_idx
        indices = np.where(mask)[0]
        if self.gene_names is not None:
            return indices, self.gene_names[indices]
        return indices

    def module_sizes(self) -> np.ndarray:
        """Number of genes per module."""
        return np.bincount(self.gene_to_module, minlength=self.n_modules)

    # ------------------------------------------------------------------
    # Fitting strategies
    # ------------------------------------------------------------------

    def _fit_correlation(self, X: np.ndarray, rng: np.random.RandomState) -> None:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        std = X.std(axis=0)
        std[std == 0] = 1.0
        X_norm = (X - X.mean(axis=0)) / std

        corr = np.corrcoef(X_norm.T)
        np.fill_diagonal(corr, 1.0)
        corr = np.nan_to_num(corr, nan=0.0)

        dist = np.clip(1.0 - np.abs(corr), 0, 2)
        np.fill_diagonal(dist, 0.0)

        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="ward")
        self.gene_to_module = fcluster(Z, t=self.n_modules, criterion="maxclust") - 1
        self._compute_module_weights(X)

    def _fit_pca(self, X: np.ndarray, rng: np.random.RandomState) -> None:
        from sklearn.decomposition import PCA

        n_comp = min(self.n_modules, X.shape[0], X.shape[1])
        pca = PCA(n_components=n_comp, random_state=self.random_state)
        pca.fit(X)

        loadings = np.abs(pca.components_)  # (n_comp, n_genes)
        self.gene_to_module = loadings.argmax(axis=0)
        self.module_weights = pca.components_
        # If fewer components than modules, map remaining
        if n_comp < self.n_modules:
            logger.warning(
                "PCA returned %d components < %d modules; "
                "excess modules will be empty.", n_comp, self.n_modules,
            )

    def _fit_kmeans(self, X: np.ndarray, rng: np.random.RandomState) -> None:
        from sklearn.cluster import MiniBatchKMeans

        kmeans = MiniBatchKMeans(
            n_clusters=self.n_modules,
            random_state=self.random_state,
            batch_size=min(1024, self.n_genes),
        )
        # Cluster genes: transpose so genes are samples
        self.gene_to_module = kmeans.fit_predict(X.T)
        self._compute_module_weights(X)

    def _fit_random(self, rng: np.random.RandomState) -> None:
        self.gene_to_module = rng.randint(0, self.n_modules, size=self.n_genes)
        self.module_weights = None

    def _compute_module_weights(self, X: np.ndarray) -> None:
        """Compute soft assignment weights from hard clustering."""
        self.module_weights = np.zeros(
            (self.n_modules, self.n_genes), dtype=np.float32,
        )
        for m in range(self.n_modules):
            mask = self.gene_to_module == m
            if not mask.any():
                continue
            gene_means = X[:, mask].mean(axis=0)
            total = gene_means.sum()
            if total > 0:
                self.module_weights[m, mask] = gene_means / total
            else:
                self.module_weights[m, mask] = 1.0 / mask.sum()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save tokeniser state to a ``.npz`` file."""
        path = Path(path)
        arrays: dict = {
            "gene_to_module": self.gene_to_module,
        }
        if self.module_weights is not None:
            arrays["module_weights"] = self.module_weights
        if self.gene_names is not None:
            arrays["gene_names"] = self.gene_names
        np.savez_compressed(path, **arrays)

        meta = {
            "n_modules": self.n_modules,
            "strategy": self.strategy,
            "random_state": self.random_state,
            "aux_tokens": self.aux_tokens,
            "n_genes": self.n_genes,
        }
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info("Saved tokeniser to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GeneModuleTokenizer":
        """Load a previously saved tokeniser."""
        path = Path(path)
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text())

        tok = cls(
            n_modules=meta["n_modules"],
            strategy=meta["strategy"],
            random_state=meta["random_state"],
            aux_tokens=meta["aux_tokens"],
        )
        tok.n_genes = meta["n_genes"]

        data = np.load(path if path.suffix == ".npz" else path.with_suffix(".npz"),
                       allow_pickle=True)
        tok.gene_to_module = data["gene_to_module"]
        tok.module_weights = data.get("module_weights")
        tok.gene_names = data.get("gene_names")
        tok._fitted = True
        return tok

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"GeneModuleTokenizer(n_modules={self.n_modules}, "
            f"strategy='{self.strategy}', {status})"
        )


# ---------------------------------------------------------------------------
# Learnable (differentiable) gene-to-module assignment
# ---------------------------------------------------------------------------

class LearnableGeneModuleLayer(nn.Module):
    """Differentiable gene-to-module aggregation layer.

    Uses a Gumbel-Softmax–parameterised assignment matrix to map a
    gene-level expression vector to a module-level representation that
    can be trained end-to-end with the student model.

    Parameters
    ----------
    n_genes : int
    n_modules : int
    temperature : float
        Gumbel-Softmax temperature (lower = harder assignments).
    """

    def __init__(
        self,
        n_genes: int,
        n_modules: int,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.n_modules = n_modules
        self.temperature = temperature

        # Learnable logits for gene → module assignment
        self.assignment_logits = nn.Parameter(
            torch.randn(n_genes, n_modules) * 0.01
        )

    def get_assignments(self, hard: bool = False) -> torch.Tensor:
        """Return (n_genes, n_modules) assignment probabilities."""
        if hard and self.training:
            return torch.nn.functional.gumbel_softmax(
                self.assignment_logits,
                tau=self.temperature,
                hard=True,
                dim=-1,
            )
        return torch.softmax(self.assignment_logits / self.temperature, dim=-1)

    def forward(self, gene_expression: torch.Tensor) -> torch.Tensor:
        """Aggregate gene expression to module expression.

        Parameters
        ----------
        gene_expression : (batch, n_genes)

        Returns
        -------
        module_expression : (batch, n_modules)
        """
        A = self.get_assignments(hard=self.training)  # (n_genes, n_modules)
        return gene_expression @ A  # (batch, n_modules)

    def to_hard_tokenizer(self, aux_tokens: int = 30) -> GeneModuleTokenizer:
        """Export as a deterministic :class:`GeneModuleTokenizer`."""
        with torch.no_grad():
            A = torch.softmax(self.assignment_logits, dim=-1)
            assignments = A.argmax(dim=-1).cpu().numpy()
            weights = A.T.cpu().numpy()  # (n_modules, n_genes)

        tok = GeneModuleTokenizer(
            n_modules=self.n_modules,
            strategy="learned",
            aux_tokens=aux_tokens,
        )
        tok.n_genes = self.n_genes
        tok.gene_to_module = assignments
        tok.module_weights = weights
        tok._fitted = True
        return tok
