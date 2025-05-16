from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

__all__ = ["compute_features", "compute_average_features"]

def compute_features(matrix: sp.spmatrix | np.ndarray):
    """Return a **dict** of structural & numerical properties of *matrix*.

    The implementation is an expanded version of your original project’s
    feature extractor.  All stats are scalar so they can be JSON‑encoded
    and dropped straight into the template.
    """

    # ––– ensure CSR –––––––––––––––––––––––––––––––––––––––––––––––––––
    if not sp.isspmatrix(matrix):
        matrix = sp.csr_matrix(matrix)
    else:
        matrix = matrix.tocsr()

    rows, cols = matrix.shape
    nnz        = matrix.nnz
    density    = nnz / (rows * cols) * 100.0 if rows and cols else 0.0

    features: dict[str, float | int | bool | None] = {
        "num_rows":           rows,
        "num_cols":           cols,
        "num_nonzeros":       nnz,
        "density_percent":    density,
    }

    # ––– symmetry –––––––––––––––––––––––––––––––––––––––––––––––––––––
    if rows == cols:
        features["pattern_symmetry"]   = (matrix != matrix.T).nnz == 0
        features["numerical_symmetry"] = (matrix - matrix.T).nnz == 0
    else:
        features["pattern_symmetry"]   = False
        features["numerical_symmetry"] = False

    # ––– NNZ per‑row / per‑col stats ––––––––––––––––––––––––––––––––
    nnz_row = matrix.getnnz(axis=1)
    nnz_col = matrix.getnnz(axis=0)

    features |= {
        "nonzeros_per_row_min": int(nnz_row.min()),
        "nonzeros_per_row_max": int(nnz_row.max()),
        "nonzeros_per_row_avg": float(nnz_row.mean()),
        "nonzeros_per_row_std": float(nnz_row.std()),

        "nonzeros_per_col_min": int(nnz_col.min()),
        "nonzeros_per_col_max": int(nnz_col.max()),
        "nonzeros_per_col_avg": float(nnz_col.mean()),
        "nonzeros_per_col_std": float(nnz_col.std()),
    }

    # ––– value stats ––––––––––––––––––––––––––––––––––––––––––––––––
    data = matrix.data if matrix.data.size else np.array([0])
    features |= {
        "value_min": float(data.min()),
        "value_max": float(data.max()),
        "value_avg": float(data.mean()),
        "value_std": float(data.std()),
    }

    # ––– per‑row detailed stats ––––––––––––––––––––––––––––––––––––
    csr = matrix
    row_min    = np.zeros(rows)
    row_max    = np.zeros(rows)
    row_mean   = np.zeros(rows)
    row_std    = np.zeros(rows)
    row_median = np.zeros(rows)

    for i in range(rows):
        seg = csr.data[csr.indptr[i] : csr.indptr[i + 1]]
        if seg.size:
            row_min[i]    = seg.min()
            row_max[i]    = seg.max()
            row_mean[i]   = seg.mean()
            row_std[i]    = seg.std()
            row_median[i] = np.median(seg)

    features |= {
        "row_min_min":   float(row_min.min()),
        "row_min_max":   float(row_min.max()),
        "row_min_mean":  float(row_min.mean()),
        "row_min_std":   float(row_min.std()),

        "row_max_min":   float(row_max.min()),
        "row_max_max":   float(row_max.max()),
        "row_max_mean":  float(row_max.mean()),
        "row_max_std":   float(row_max.std()),

        "row_mean_min":  float(row_mean.min()),
        "row_mean_max":  float(row_mean.max()),
        "row_mean_mean": float(row_mean.mean()),
        "row_mean_std":  float(row_mean.std()),

        "row_std_min":   float(row_std.min()),
        "row_std_max":   float(row_std.max()),
        "row_std_mean":  float(row_std.mean()),
        "row_std_std":   float(row_std.std()),

        "row_median_min":   float(row_median.min()),
        "row_median_max":   float(row_median.max()),
        "row_median_mean":  float(row_median.mean()),
        "row_median_std":   float(row_median.std()),
    }

    # ––– per‑column detailed stats ––––––––––––––––––––––––––––––––––
    csc = matrix.tocsc()
    col_min    = np.zeros(cols)
    col_max    = np.zeros(cols)
    col_mean   = np.zeros(cols)
    col_std    = np.zeros(cols)
    col_median = np.zeros(cols)

    for j in range(cols):
        seg = csc.data[csc.indptr[j] : csc.indptr[j + 1]]
        if seg.size:
            col_min[j]    = seg.min()
            col_max[j]    = seg.max()
            col_mean[j]   = seg.mean()
            col_std[j]    = seg.std()
            col_median[j] = np.median(seg)

    features |= {
        "col_min_min":   float(col_min.min()),
        "col_min_max":   float(col_min.max()),
        "col_min_mean":  float(col_min.mean()),
        "col_min_std":   float(col_min.std()),

        "col_max_min":   float(col_max.min()),
        "col_max_max":   float(col_max.max()),
        "col_max_mean":  float(col_max.mean()),
        "col_max_std":   float(col_max.std()),

        "col_mean_min":  float(col_mean.min()),
        "col_mean_max":  float(col_mean.max()),
        "col_mean_mean": float(col_mean.mean()),
        "col_mean_std":  float(col_mean.std()),

        "col_std_min":   float(col_std.min()),
        "col_std_max":   float(col_std.max()),
        "col_std_mean":  float(col_std.mean()),
        "col_std_std":   float(col_std.std()),

        "col_median_min":   float(col_median.min()),
        "col_median_max":   float(col_median.max()),
        "col_median_mean":  float(col_median.mean()),
        "col_median_std":   float(col_median.std()),
    }

    # ––– diagonal distances / bandwidth ––––––––––––––––––––––––––––
    row_idx, col_idx = matrix.nonzero()
    if row_idx.size:
        dist = np.abs(row_idx - col_idx)
        features["avg_distance_to_diagonal"]         = float(dist.mean())
        features["bandwidth"]                        = int(dist.max())
        features["num_diagonals_with_nonzeros"]     = int(np.unique(dist).size)
    else:
        features["avg_distance_to_diagonal"]         = 0.0
        features["bandwidth"]                        = 0
        features["num_diagonals_with_nonzeros"]     = 0

    # ––– structural unsymmetry ––––––––––––––––––––––––––––––––––––
    unsym = set(zip(row_idx, col_idx)) - set(zip(col_idx, row_idx))
    features["num_structurally_unsymmetric_elements"] = int(len(unsym))

    # ––– norms ––––––––––––––––––––––––––––––––––––––––––––––––––––
    try:
        features["norm_1"]         = float(splinalg.norm(matrix, 1))
        features["norm_inf"]       = float(splinalg.norm(matrix, np.inf))
        features["frobenius_norm"] = float(splinalg.norm(matrix))
    except Exception:
        pass

    # ––– 1‑norm condition estimate ––––––––––––––––––––––––––––––––
    try:
        features["estimated_condition_number"] = float(splinalg.onenormest(matrix))
    except Exception:
        features["estimated_condition_number"] = None

    return features

def compute_average_features(features_list: list[dict]):
    """Element‑wise average of numeric entries across *features_list*."""
    if not features_list:
        return {}

    avg: dict[str, float | int | bool | None] = {}
    keys = features_list[0].keys()
    for k in keys:
        v0 = features_list[0][k]
        if isinstance(v0, (int, float, np.number, bool)):
            avg[k] = float(np.mean([f[k] for f in features_list]))
        else:
            avg[k] = v0
    return avg