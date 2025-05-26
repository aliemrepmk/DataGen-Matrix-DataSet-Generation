import numpy as np
from scipy.sparse import coo_matrix

def expand_matrix(original_matrix, desired_rows, additional_density=2):
    """
    Expand a sparse matrix by scaling original non-zero positions proportionally,
    preserving the pattern, and increasing non-zero count with controlled density.

    Parameters:
    - original_matrix: scipy.sparse matrix (e.g., COO or CSR).
    - desired_rows: Number of rows in the expanded matrix.
    - additional_density: Number of new non-zeros to add around each scaled position.

    Returns:
    - expanded_matrix: A scipy.sparse.coo_matrix with increased non-zeros.
    """
    desired_cols = desired_rows  # square matrix

    # Convert to COO format for easy manipulation
    if not isinstance(original_matrix, coo_matrix):
        original_matrix = original_matrix.tocoo()

    orig_rows, orig_cols = original_matrix.shape

    row_scale = desired_rows / orig_rows
    col_scale = desired_cols / orig_cols

    data = []
    rows = []
    cols = []

    # Value range for new entries
    min_val = original_matrix.data.min()
    max_val = original_matrix.data.max()

    for i in range(len(original_matrix.data)):
        old_r = original_matrix.row[i]
        old_c = original_matrix.col[i]
        val = original_matrix.data[i]

        # Scale position
        new_r = min(int(old_r * row_scale), desired_rows - 1)
        new_c = min(int(old_c * col_scale), desired_cols - 1)

        rows.append(new_r)
        cols.append(new_c)
        data.append(val)

        # Add additional random non-zeros around the scaled position
        for _ in range(additional_density):
            jitter_r = np.clip(new_r + np.random.randint(-3, 4), 0, desired_rows - 1)
            jitter_c = np.clip(new_c + np.random.randint(-3, 4), 0, desired_cols - 1)
            rows.append(jitter_r)
            cols.append(jitter_c)
            data.append(np.random.uniform(min_val, max_val))

    expanded_matrix = coo_matrix((data, (rows, cols)), shape=(desired_rows, desired_cols))
    return expanded_matrix.tocsr()  # Convert to CSR format for efficient operations
