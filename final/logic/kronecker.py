"""
logic.kronecker
~~~~~~~~~~~~~~~~
Scale a sparse matrix up or down by *learning* a small stochastic
Kronecker *initiator* from the input graph and then expanding it via
Kronecker products.  This module plugs into your existing Flask app
under the same API contract as your other scaling routines.

Key design points
-----------------
* **No heavyweight deps** – only ``numpy`` + ``scipy`` are required.
* **Sparse‑safe** – we never densify the full matrix; densities are
  matched probabilistically.
* **Density‑aware initiator scaling** – the initiator is renormalised so
  that its Kronecker expansion has (in expectation) the **same density**
  as the original graph.  This prevents the zero‑edge issue you hit in
  early tests.

Usage
-----
```python
from logic.load_matrix import load_matrix
from logic.kronecker import scale_sparse_matrix_kronecker

A = load_matrix("uploaded-matrices/cage7.mtx")
scaled = scale_sparse_matrix_kronecker(A, 1015, "generated/cage7_kron.mtx")
```
"""
from __future__ import annotations

from typing import Tuple
import math
import random

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.io import mmwrite

__all__ = [
    "fit_kronecker_initiator",
    "kronecker_expand_initiator",
    "scale_sparse_matrix_kronecker",
]

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _partition_indices(n: int, blocks: int) -> list[Tuple[int, int]]:
    """Split ``range(n)`` into ``blocks`` nearly‑equal slices."""
    base = n // blocks
    extra = n % blocks
    indices: list[Tuple[int, int]] = []
    start = 0
    for b in range(blocks):
        stop = start + base + (1 if b < extra else 0)
        indices.append((start, stop))
        start = stop
    return indices


# -----------------------------------------------------------------------------
# 1.  Fit a *stochastic* initiator matrix
# -----------------------------------------------------------------------------

def fit_kronecker_initiator(
    matrix: sparse.spmatrix,
    initiator_size: int = 2,
    eps: float = 1e-6,
) -> NDArray[np.float64]:
    """Return a dense ``initiator_size×initiator_size`` probability matrix.

    A simple but surprisingly strong baseline: **block‑average** the
    adjacency mask and clip to ``[eps, 1‑eps]``.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Kronecker method requires a *square* matrix.")

    n = matrix.shape[0]
    mask = matrix.astype(bool)
    mask.eliminate_zeros()

    rows = _partition_indices(n, initiator_size)
    cols = rows  # square

    init = np.zeros((initiator_size, initiator_size), dtype=float)
    for i, (rs, re) in enumerate(rows):
        for j, (cs, ce) in enumerate(cols):
            block = mask[rs:re, cs:ce]
            total = (re - rs) * (ce - cs)
            init[i, j] = block.nnz / total if total else 0.0

    return np.clip(init, eps, 1.0 - eps)


# -----------------------------------------------------------------------------
# 2.  Kronecker expansion utilities
# -----------------------------------------------------------------------------

def _num_expansions(target_size: int, initiator_size: int) -> int:
    """Minimum ``k`` s.t. ``initiator_size**k >= target_size``."""
    return math.ceil(math.log(target_size, initiator_size))


def _renormalise_initiator(
    initiator: NDArray[np.float64],
    target_density: float,
    expansions: int,
    eps: float = 1e-6,
) -> NDArray[np.float64]:
    """Scale initiator so that ``mean(init)**expansions ≈ target_density``."""
    current_mean = initiator.mean()
    desired_mean = target_density ** (1 / expansions)
    if current_mean == 0:
        factor = 1.0
    else:
        factor = desired_mean / current_mean
    scaled = np.clip(initiator * factor, eps, 1.0 - eps)
    return scaled


def kronecker_expand_initiator(
    initiator: NDArray[np.float64],
    expansions: int,
    rng: np.random.Generator,
) -> sparse.csr_matrix:
    """Return a CSR adjacency matrix after ``expansions`` Kronecker products."""
    kron_prob = initiator.copy()
    for _ in range(expansions - 1):
        kron_prob = np.kron(kron_prob, initiator)

    # Sample edges (Bernoulli) using vectorised RNG
    rand = rng.random(kron_prob.shape)
    adj_dense = (rand < kron_prob).astype(np.int8)
    adj_csr = sparse.csr_matrix(adj_dense)
    adj_csr.eliminate_zeros()
    return adj_csr


# -----------------------------------------------------------------------------
# 3.  Public API – scale_sparse_matrix_kronecker
# -----------------------------------------------------------------------------

def scale_sparse_matrix_kronecker(
    original_matrix: sparse.spmatrix,
    new_size: int,
    output_path: str,
    match_nnz: bool = True,
    *,
    initiator_size: int = 2,
    random_seed: int | None = 0,
) -> sparse.csr_matrix:
    """Scale ``original_matrix`` to ``new_size`` × ``new_size`` with a hybrid
    Kronecker graph model.
    """
    if original_matrix.shape[0] != original_matrix.shape[1]:
        raise ValueError("Original matrix must be square for Kronecker scaling.")

    rng = np.random.default_rng(random_seed)

    # 1️⃣  Fit initiator via block‑averaging
    initiator = fit_kronecker_initiator(original_matrix, initiator_size)

    # 2️⃣  Work out how many Kronecker products we’ll need
    k = _num_expansions(new_size, initiator_size)

    # 3️⃣  Density renormalisation so that E[density] matches original
    orig_density = original_matrix.nnz / (original_matrix.shape[0] ** 2)
    initiator = _renormalise_initiator(initiator, orig_density, k)

    # 4️⃣  Expand
    expanded = kronecker_expand_initiator(initiator, k, rng)

    # 5️⃣  Crop to exact target size if overshot
    if expanded.shape[0] > new_size:
        expanded = expanded[:new_size, :new_size].tocsr()
        expanded.eliminate_zeros()

    # 6️⃣  Optional nnz matching for exact sparsity parity
    if match_nnz:
        target_nnz = int(orig_density * (new_size ** 2))
        current_nnz = expanded.nnz
        if current_nnz == 0:
            current_nnz = 1  # avoid division by zero later
        if current_nnz > target_nnz:
            # Randomly drop surplus edges
            keep_idx = rng.choice(current_nnz, size=target_nnz, replace=False)
            rows, cols = expanded.nonzero()
            chosen = set(zip(rows[keep_idx], cols[keep_idx]))
            data = np.ones(target_nnz, dtype=int)
            r, c = zip(*chosen)
            expanded = sparse.csr_matrix((data, (r, c)), shape=(new_size, new_size))
        elif current_nnz < target_nnz:
            # Randomly add edges without duplicating
            rows, cols = expanded.nonzero()
            occupied = set(zip(rows.tolist(), cols.tolist()))
            needed = target_nnz - current_nnz
            add_rows = rng.integers(0, new_size, size=needed * 2)
            add_cols = rng.integers(0, new_size, size=needed * 2)
            added = []
            for r, c in zip(add_rows, add_cols):
                if len(added) >= needed:
                    break
                if (r, c) not in occupied:
                    occupied.add((r, c))
                    added.append((r, c))
            if added:
                ar, ac = zip(*added)
                data = np.ones(len(added), dtype=int)
                extra = sparse.csr_matrix((data, (ar, ac)), shape=(new_size, new_size))
                expanded += extra
                expanded.eliminate_zeros()

    # 7️⃣  Persist & return
    mmwrite(output_path, expanded)
    return expanded
