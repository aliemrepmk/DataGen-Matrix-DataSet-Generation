import numpy as np
from scipy import sparse

def pad_to_shape(M: sparse.spmatrix, target_shape):
    """Return M zero-padded up to `target_shape` (rows, cols)."""
    r, c = M.shape
    R, C = target_shape
    if (r, c) == (R, C):
        return M              # already the right size
    # Pad rows
    if r < R:
        M = sparse.vstack([M, sparse.csr_matrix((R - r, c))], format="csr")
    # Pad cols
    if c < C:
        M = sparse.hstack([M, sparse.csr_matrix((R, C - c))], format="csr")
    return M.tocsr()

def cosine_value(A: sparse.spmatrix, B: sparse.spmatrix) -> float:
    """Cosine similarity of matrices viewed as long numeric vectors."""
    # Pad to same shape
    target_shape = (max(A.shape[0], B.shape[0]), max(A.shape[1], B.shape[1]))
    A = pad_to_shape(A, target_shape)
    B = pad_to_shape(B, target_shape)
    num   = A.multiply(B).sum()                          # ⟨A,B⟩_F
    denom = np.sqrt(A.multiply(A).sum() * B.multiply(B).sum())
    return float(num / denom) if denom != 0 else 0.0

def cosine_pattern(A: sparse.spmatrix, B: sparse.spmatrix) -> float:
    """Cosine similarity of sparsity *patterns* (binary masks)."""
    # Pad to same shape
    target_shape = (max(A.shape[0], B.shape[0]), max(A.shape[1], B.shape[1]))
    A = pad_to_shape(A, target_shape)
    B = pad_to_shape(B, target_shape)
    A_mask = A.astype(bool)
    B_mask = B.astype(bool)
    inter  = A_mask.multiply(B_mask).nnz                 # |S_A ∩ S_B|
    denom  = np.sqrt(A_mask.nnz * B_mask.nnz)
    return float(inter / denom) if denom != 0 else 0.0

def cosine_value_overlap(A, B):
    min_rows = min(A.shape[0], B.shape[0])
    min_cols = min(A.shape[1], B.shape[1])
    A_crop = A[:min_rows, :min_cols]
    B_crop = B[:min_rows, :min_cols]
    num   = A_crop.multiply(B_crop).sum()
    denom = np.sqrt(A_crop.multiply(A_crop).sum() * B_crop.multiply(B_crop).sum())
    return float(num / denom) if denom != 0 else 0.0

def cosine_pattern_overlap(A, B):
    min_rows = min(A.shape[0], B.shape[0])
    min_cols = min(A.shape[1], B.shape[1])
    A_mask = A[:min_rows, :min_cols].astype(bool)
    B_mask = B[:min_rows, :min_cols].astype(bool)
    inter  = A_mask.multiply(B_mask).nnz
    denom  = np.sqrt(A_mask.nnz * B_mask.nnz)
    return float(inter / denom) if denom != 0 else 0.0 