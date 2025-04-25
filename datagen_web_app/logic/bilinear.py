import random
import numpy as np
import scipy.io as sio
from scipy import sparse

def bilinear_interpolation(csr_matrix: sparse.csr_matrix, row: int, col: int, scale_factor: float, original_size: int):
    """
    Perform bilinear interpolation at a given (row, col) location using 4 neighboring points.

    Parameters:
    -----------
    csr_matrix : scipy.sparse.csr_matrix
        The original sparse matrix in CSR format for efficient access
    row : int
        Row index in the new (target) matrix space
    col : int
        Column index in the new (target) matrix space
    scale_factor : float
        Ratio of new_size / original_size for coordinate transformation
    original_size : int
        Size of the original matrix (assumes square matrix)

    Returns:
    --------
    float
        Interpolated value at the given coordinate using bilinear interpolation.
        Returns 0 if all contributing neighbors are zero.
    """
    # Map new matrix coordinate to location in original matrix
    orig_row_float = row / scale_factor
    orig_col_float = col / scale_factor

    # Identify four surrounding neighbors in original matrix    
    row_low = int(np.floor(orig_row_float))
    row_high = min(row_low + 1, original_size - 1)
    col_low = int(np.floor(orig_col_float))
    col_high = min(col_low + 1, original_size - 1)

    # Compute weights based on relative position within the cell
    w_row = orig_row_float - row_low
    w_col = orig_col_float - col_low

    # Retrieve the four corner values from the original sparse matrix
    val_ll = csr_matrix[row_low, col_low]   # lower-left
    val_lh = csr_matrix[row_low, col_high]  # lower-right
    val_hl = csr_matrix[row_high, col_low]  # upper-left
    val_hh = csr_matrix[row_high, col_high] # upper-right

    # Linearly interpolate between columns and rows
    top = (1 - w_col) * val_ll + w_col * val_lh
    bottom = (1 - w_col) * val_hl + w_col * val_hh
    return (1 - w_row) * top + w_row * bottom

def scale_sparse_matrix_bilinear(original_matrix: sparse.csr_matrix, new_size: int, output_path: str, match_nnz = True) -> sparse.csr_matrix:
    """
    Scale a sparse matrix with bilinear interpolation while maintaining the sparsity pattern,
    working directly with sparse representation to avoid dense conversion.
    
    Parameters:
    -----------
    original_matrix : scipy.sparse.spmatrix
        Input sparse matrix to be scaled
    new_size : int
        New size for the matrix (will be scaled to new_size x new_size)
    output_path : str
        Output path for saving the scaled matrix as .mtx file
    match_nnz: bool
        To decide match the exact number of nonzeros or not
        
    Returns:
    --------
    scipy.sparse.csr_matrix
        Scaled sparse matrix with preserved value range and linearly scaled number of nonzeros
    """
    # Convert to CSR for efficient row slicing
    original_csr = sparse.csr_matrix(original_matrix)
    
    # Get original dimensions and values
    original_size = max(original_matrix.shape)
    orig_nnz = original_matrix.nnz
    
    # Calculate scaling factor and target nonzeros
    scale_factor = new_size / original_size
    target_nnz = int(orig_nnz * scale_factor)
    
    # Create a dictionary to store new coordinates and values
    new_coords = {}
    
    # Calculate how many points to sample initially
    sample_count = int(target_nnz * 1.5)
    
    # Generate random coordinates in the new matrix
    candidates = set()
    while len(candidates) < sample_count:
        new_row = random.randint(0, new_size - 1)
        new_col = random.randint(0, new_size - 1)
        candidates.add((new_row, new_col))
    
    # For each sampled coordinate, perform bilinear interpolation
    for new_row, new_col in candidates:
        value = bilinear_interpolation(original_csr, new_row, new_col, scale_factor, original_size)
        # Only add non-zero values to our result
        if abs(value) > 1e-10:
            new_coords[(new_row, new_col)] = value
    
    # Ensure to have exactly target_nnz non-zeros if required
    if match_nnz:
        current_coords = list(new_coords.items())
        current_nnz = len(current_coords)
        
        if current_nnz > target_nnz:
            # Too many non-zeros, randomly remove some
            new_coords = dict(random.sample(current_coords, target_nnz))
        elif current_nnz < target_nnz:
            # Too few non-zeros, add more
            while len(new_coords) < target_nnz:
                new_row = random.randint(0, new_size - 1)
                new_col = random.randint(0, new_size - 1)
                
                if (new_row, new_col) not in new_coords:
                    value = bilinear_interpolation(original_csr, new_row, new_col, scale_factor, original_size)
                    if abs(value) > 1e-10:
                        new_coords[(new_row, new_col)] = value
    
    # Convert the dictionary to COO format
    result_rows = []
    result_cols = []
    result_vals = []
    
    for (row, col), val in new_coords.items():
        result_rows.append(row)
        result_cols.append(col)
        result_vals.append(val)
    
    # Create the final sparse matrix and save it to output_path
    scaled_matrix = sparse.csr_matrix((result_vals, (result_rows, result_cols)), shape=(new_size, new_size))
    sio.mmwrite(output_path, scaled_matrix)
    
    return scaled_matrix