import numpy as np
from scipy.fft import dctn, idctn
import scipy.sparse as sp

def resize_sparse_dct(X_sparse: sp.spmatrix, new_shape: tuple[int, int], thresh):
    """
    Resize a 2-D **sparse** array with DCT-domain crop / pad.

    Parameters
    ----------
    X_sparse : scipy.sparse.spmatrix
        Input matrix (any sparse format).
    new_shape : tuple[int, int]
        Desired height and width (rows, cols).
    thresh : float, default 1e-8
        Magnitude below which coefficients are zeroed to
        recover sparsity after the inverse DCT.

    Returns
    -------
    Y_sparse : scipy.sparse.csr_matrix
        Resized matrix in CSR format.
    """
    if not sp.issparse(X_sparse):
        raise TypeError("Input must be a SciPy sparse matrix")

    # -- DCT (dense) ---------------------------------------------------------
    X = X_sparse.toarray()                       # one dense copy
    F = dctn(X, norm='ortho')                    # forward 2-D DCT-II

    # -- frequency-domain crop / pad ----------------------------------------
    new_rows, new_cols = new_shape
    target_F = np.zeros(new_shape, dtype=F.dtype)
    copy_rows = min(new_rows, F.shape[0])
    copy_cols = min(new_cols, F.shape[1])
    target_F[:copy_rows, :copy_cols] = F[:copy_rows, :copy_cols]

    # -- inverse DCT ---------------------------------------------------------
    Y = idctn(target_F, norm='ortho')            # inverse 2-D DCT-III

    # -- sparsify ------------------------------------------------------------
    Y[np.abs(Y) < thresh] = 0.0                  # hard-threshold small values
    Y_sparse = sp.csr_matrix(Y)                     # back to sparse

    return Y_sparse

def scale_sparse_matrix_dct_blockwise(original_matrix: sp.csr_matrix, new_size: int, block_size: int = 8, thresh: float = 1e-8) -> sp.csr_matrix:
    """
    Resize a sparse matrix using blockwise DCT to a square matrix of size new_size x new_size.
    Only processes nonzero blocks.

    Parameters
    ----------
    original_matrix : scipy.sparse.csr_matrix
        Input sparse matrix to be resized
    new_size : int
        Desired size for both dimensions of the output square matrix
    block_size : int, default=8
        Size of blocks to process in DCT domain
    thresh : float, default=1e-8
        Threshold for zeroing small values after inverse DCT

    Returns
    -------
    scipy.sparse.csr_matrix
        Resized square matrix of size new_size x new_size
    """
    orig_rows, orig_cols = original_matrix.shape
    scale = new_size / max(orig_rows, orig_cols)  # Use max to maintain aspect ratio

    rows, cols = original_matrix.nonzero()
    values = original_matrix.data

    blocks = {}
    for i, (row, col) in enumerate(zip(rows, cols)):
        block_row = row // block_size
        block_col = col // block_size
        block_key = (block_row, block_col)
        if block_key not in blocks:
            blocks[block_key] = {'positions': [], 'values': []}
        rel_row = row % block_size
        rel_col = col % block_size
        blocks[block_key]['positions'].append((rel_row, rel_col))
        blocks[block_key]['values'].append(values[i])

    result_data = []
    result_rows = []
    result_cols = []

    for (block_row, block_col), block_data in blocks.items():
        block_dense = np.zeros((block_size, block_size))
        for (rel_row, rel_col), value in zip(block_data['positions'], block_data['values']):
            block_dense[rel_row, rel_col] = value

        # Calculate new block dimensions using the same scale for both dimensions
        new_block_size = max(1, int(round(block_size * scale)))

        # Skip if block is empty
        if np.all(block_dense == 0):
            continue

        # DCT
        F = dctn(block_dense, norm='ortho')

        # Resize in frequency domain (crop/pad)
        target_F = np.zeros((new_block_size, new_block_size), dtype=F.dtype)
        copy_size = min(new_block_size, F.shape[0], F.shape[1])
        target_F[:copy_size, :copy_size] = F[:copy_size, :copy_size]

        # Inverse DCT
        block_resized = idctn(target_F, norm='ortho')

        # Threshold and store nonzeros
        for r in range(block_resized.shape[0]):
            for c in range(block_resized.shape[1]):
                val = block_resized[r, c]
                if abs(val) > thresh:
                    row_idx = int(block_row * block_size * scale) + r
                    col_idx = int(block_col * block_size * scale) + c
                    if row_idx < new_size and col_idx < new_size:
                        result_rows.append(row_idx)
                        result_cols.append(col_idx)
                        result_data.append(val)

    return sp.csr_matrix((result_data, (result_rows, result_cols)), shape=(new_size, new_size))