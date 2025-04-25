import numpy as np
import scipy.io as sio
from scipy import sparse
import random
from collections import defaultdict

# ------------------------------------------------------------------
# -------------  ONE–STEP HEAVY-EDGE-MATCH COARSENING --------------
# ------------------------------------------------------------------
def heavy_edge_matching(adj: sparse.csr_matrix):
    """
    One pass of Heavy-Edge Matching.
    Returns:
        parents  – np.ndarray, len = n, gives the coarse node index for each fine node
        coarse_n – number of coarse nodes produced
    """
    n = adj.shape[0]
    parents = -np.ones(n, dtype=np.int32)
    coarse_id = 0

    for u in range(n):
        if parents[u] != -1:
            continue                      # already matched
        # find heaviest neighbor that is still unmatched
        start, end = adj.indptr[u], adj.indptr[u+1]
        neighs = adj.indices[start:end]
        weights = adj.data[start:end]
        if len(neighs) == 0:
            parents[u] = coarse_id        # isolated vertex
            coarse_id += 1
            continue

        # pick unmatched neighbor with max weight
        mask = parents[neighs] == -1
        if mask.any():
            v = neighs[mask][np.argmax(weights[mask])]
            parents[[u, v]] = coarse_id
        else:
            parents[u] = coarse_id
        coarse_id += 1

    return parents, coarse_id


def coarsen_once(adj: sparse.csr_matrix):
    """Apply one HEM pass and build the coarse adjacency matrix."""
    parents, coarse_n = heavy_edge_matching(adj)
    # Build triple lists for COO
    row, col = adj.nonzero()
    data = adj.data
    new_row = parents[row]
    new_col = parents[col]

    # combine duplicate edges by summing weights
    hash_idx = new_row * coarse_n + new_col
    order = np.argsort(hash_idx)
    hash_idx = hash_idx[order]
    data_sorted = data[order]

    uniq, first = np.unique(hash_idx, return_index=True)
    sums = np.add.reduceat(data_sorted, first)
    row_c = uniq // coarse_n
    col_c = uniq %  coarse_n
    coarse = sparse.coo_matrix((sums, (row_c, col_c)), shape=(coarse_n, coarse_n))

    # keep symmetric structure & remove self-loops
    coarse = coarse.tocsr()
    coarse.setdiag(0)
    coarse.eliminate_zeros()
    coarse = 0.5 * (coarse + coarse.T)
    return coarse.tocsr(), parents


# ------------------------------------------------------------------
# -----------  WEIGHT-AWARE  REFINEMENT / PROLONGATION -------------
# ------------------------------------------------------------------
def refine_once(adj_coarse: sparse.csr_matrix,
                splits: int = 2,
                rng: np.random.Generator | None = None):
    """
    Duplicate each coarse edge, but the **number of duplicates**
    (i.e. how many of the `splits²` possible child-edges we actually
    keep) is proportional to the original weight:

        high weight  →  keep most duplicates
        low  weight  →  keep few duplicates

    Every kept duplicate carries           weight = orig_val / k,
    so the total weight per coarse edge is preserved in expectation.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_c         = adj_coarse.shape[0]
    rows, cols  = adj_coarse.nonzero()
    vals        = adj_coarse.data
    vmax        = vals.max() if vals.size else 1.0

    row_out, col_out, val_out = [], [], []

    # iterate over each coarse edge once
    for r, c, w in zip(rows, cols, vals):
        # normalise weight → in (0,1]; use it as duplication probability
        p      = min(1.0, (w / vmax) ** 0.2 )          # high w ⇒ p≈1, low w ⇒ p≪1 1 burda alpha değeri alpha >1 sharper 
        choices = [(i, j)
                   for i in range(splits)
                   for j in range(splits)]
        kept    = [ (i, j) for (i, j) in choices if rng.random() < p ]
        if not kept:                          # always keep at least one
            kept = [choices[rng.integers(len(choices))]]
        k       = len(kept)                   # how many duplicates kept
        w_child = w / k                       # preserve total weight

        # create duplicates
        for i, j in kept:
            row_out.append(r * splits + i)
            col_out.append(c * splits + j)
            val_out.append(w_child)

    n_f = n_c * splits
    fine = sparse.coo_matrix((val_out, (row_out, col_out)),
                             shape=(n_f, n_f)).tocsr()
    fine.setdiag(0)
    fine.eliminate_zeros()
    return 0.5 * (fine + fine.T)


# ------------------------------------------------------------------
# --------  WEIGHT-AWARE NODE SAMPLING  (final trim step) ----------
# ------------------------------------------------------------------
def _sample_nodes_by_strength(A: sparse.csr_matrix,
                              new_size: int,
                              rng: np.random.Generator | None = None):
    """
    Return a sub-matrix of `A` with exactly `new_size` rows/cols.
    Nodes are chosen *without replacement* with probability
        p_i ∝ strength(i) = Σ_j |A_ij|.
    """
    if rng is None:
        rng = np.random.default_rng()

    # node strengths (degree for adjacency, row-sum of weights)
    strength = np.asarray(A.sum(axis=1)).ravel()
    # avoid zeros → give tiny probability so np.random.choice won't complain
    strength = strength + 1e-12
    probs    = strength / strength.sum()

    keep = rng.choice(A.shape[0], size=new_size, replace=False, p=probs)
    keep.sort()
    return A[keep][:, keep].tocsr()
# ------------------------------------------------------------------
# ------------------  TOP-LEVEL SCALE FUNCTION ---------------------
# ------------------------------------------------------------------
def scale_sparse_matrix_graph(original_matrix: sparse.csr_matrix,
                              new_size: int,
                              output_path: str,
                              match_nnz: bool = True):
    """
    Scale an adjacency/Laplacian matrix via multilevel graph coarsening & refinement.
    Mirrors the signature of your other scalers.
    """
    A = original_matrix.tocsr()
    orig_n = A.shape[0]

# ---------- DOWN-SCALE (coarsen) ----------
    if new_size < orig_n:
    # 1) HEM passes until next would overshoot
        while A.shape[0] // 2 >= new_size:
            A, _ = coarsen_once(A)

    # 2) weight-aware sampling if still above target
    if A.shape[0] > new_size:
        A = _sample_nodes_by_strength(A, new_size)

    # 2) if still bigger than target, random-sample nodes
    if A.shape[0] > new_size:
        keep = np.random.choice(A.shape[0], new_size, replace=False)
        keep.sort()
        A = A[keep][:, keep].tocsr()

    # ---------- UP-SCALE (refine) ------------
    elif new_size > orig_n:
        target = new_size
        while A.shape[0] < target:
            # decide split factor: 2 is simple; use 3 if that lands closer
            remaining = target / A.shape[0]
            split = 3 if remaining > 2.5 else 2
            A = refine_once(A, splits=split)
    current_nnz = A.nnz
    desired_nnz = int(original_matrix.nnz * (new_size / orig_n))
    # ---------- OPTIONAL nnz MATCHING ----------
    if match_nnz:

        if current_nnz > desired_nnz:
            # 🛠 Fix: convert to COO first
            A = A.tocoo()
            keep = np.random.choice(current_nnz, size=desired_nnz, replace=False)
            A = sparse.coo_matrix((A.data[keep], (A.row[keep], A.col[keep])), shape=A.shape).tocsr()
        elif current_nnz < desired_nnz:
            # sprinkle small weights at random vacant positions
            extra_needed = desired_nnz - current_nnz
            rng = np.random.default_rng()
            rows = rng.integers(0, new_size, size=extra_needed)
            cols = rng.integers(0, new_size, size=extra_needed)
            mask = rows != cols
            rows, cols = rows[mask], cols[mask]
            vals = np.full(len(rows), A.data.mean() if A.nnz else 1.0)
            A_extra = sparse.coo_matrix((vals, (rows, cols)),
                                        shape=A.shape)
            A = (A + A_extra).tocsr()
            A.setdiag(0)
            A.eliminate_zeros()

    # ---------- SAVE & RETURN ----------
    sio.mmwrite(output_path, A)
    return A
