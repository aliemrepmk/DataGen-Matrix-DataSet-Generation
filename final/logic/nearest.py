import random
import numpy as np
import scipy.sparse as sp

def generate_neighbor_point(base_row, base_col, half_size, new_size):
    """
    Generate a new point in the neighborhood of a base point.
    
    Parameters:
    -----------
    base_row : int
        Row coordinate of base point
    base_col : int
        Column coordinate of base point
    half_size : int
        Half the size of the neighborhood to consider
    new_size : int
        Size of the matrix (to enforce bounds)
        
    Returns:
    --------
    tuple
        (new_row, new_col) coordinates of the generated point
    """
    # Generate random offsets within the neighborhood
    row_offset = random.randint(-half_size, half_size)
    col_offset = random.randint(-half_size, half_size)
    
    # Apply offset and ensure coordinates stay within matrix bounds
    new_row = min(max(0, base_row + row_offset), new_size - 1)
    new_col = min(max(0, base_col + col_offset), new_size - 1)
    
    return new_row, new_col

def generate_additional_points(base_rows, base_cols, count, half_size, new_size, orig_vals):
    """
    Generate additional non-zero points around existing points.
    
    Parameters:
    -----------
    base_rows : array_like
        Row coordinates of base points
    base_cols : array_like
        Column coordinates of base points
    count : int
        Number of additional points to generate
    half_size : int
        Half the size of the neighborhood to consider
    new_size : int
        Size of the matrix (to enforce bounds)
    orig_vals : array_like
        Original values to sample from for new points
        
    Returns:
    --------
    tuple
        (new_rows, new_cols, new_vals) arrays containing coordinates and values of new points
    """
    base_count = len(base_rows)
    
    # Randomly select some existing nonzeros to generate neighbors for
    indices_for_neighbors = np.random.choice(base_count, size=count, replace=True)
    
    # Pre-allocate arrays for efficiency
    additional_rows = np.zeros(count, dtype=int)
    additional_cols = np.zeros(count, dtype=int)
    additional_vals = np.zeros(count, dtype=orig_vals.dtype)

    # Generate new nonzeros near selected existing ones
    for i, idx in enumerate(indices_for_neighbors):
        base_row = base_rows[idx]
        base_col = base_cols[idx]

        # Generate neighbor using our helper function
        additional_rows[i], additional_cols[i] = generate_neighbor_point(
            base_row, base_col, half_size, new_size
        )
        additional_vals[i] = np.random.choice(orig_vals)
    
    return additional_rows, additional_cols, additional_vals

def handle_collisions(rows, cols, vals, new_size):
    """
    Handle overlapping points by keeping the maximum value at each position.
    
    Parameters:
    -----------
    rows : array_like
        Row coordinates of points
    cols : array_like
        Column coordinates of points
    vals : array_like
        Values of points
    new_size : int
        Size of the matrix
        
    Returns:
    --------
    tuple
        (unique_rows, unique_cols, unique_vals) arrays with collisions resolved
    """
    # Convert 2D indices to unique 1D positions
    positions = rows * new_size + cols
    unique_positions, inverse_indices = np.unique(positions, return_inverse=True)
    
    # For each unique position, find the max value among duplicates
    result_vals = np.zeros(len(unique_positions), dtype=vals.dtype)
    for i, val in enumerate(vals):
        pos_idx = inverse_indices[i]
        result_vals[pos_idx] = max(result_vals[pos_idx], val) if result_vals[pos_idx] != 0 else val
    
    # Convert unique positions back to row, col format
    result_rows = unique_positions // new_size
    result_cols = unique_positions % new_size
    
    return result_rows, result_cols, result_vals

def add_points(base_rows, base_cols, base_vals, additional_needed, half_size, new_size, orig_vals):
    """
    Parameters:
    -----------
    base_rows : array_like
        Row coordinates of existing points
    base_cols : array_like
        Column coordinates of existing points
    base_vals : array_like
        Values of existing points
    additional_needed : int
        Number of additional points to add
    half_size : int
        Half the size of the neighborhood to consider
    new_size : int
        Size of the matrix
    orig_vals : array_like
        Original values to sample from for new points
        
    Returns:
    --------
    tuple
        (combined_rows, combined_cols, combined_vals) arrays containing all points
    """
    current_nnz = len(base_rows)
    
    # Pre-allocate arrays for the new points
    add_rows = np.zeros(additional_needed, dtype=int)
    add_cols = np.zeros(additional_needed, dtype=int)
    add_vals = np.zeros(additional_needed, dtype=orig_vals.dtype)
    
    # Create a set of existing positions for faster lookup
    existing_positions = set(zip(base_rows, base_cols))
    
    i = 0
    while i < additional_needed:
        # Choose a random existing nonzero to add a neighbor
        idx = random.randint(0, current_nnz - 1)
        base_row = base_rows[idx]
        base_col = base_cols[idx]

        # Generate neighbor using our helper function
        new_row, new_col = generate_neighbor_point(base_row, base_col, half_size, new_size)
        
        # Check if position is already occupied
        new_pos = (new_row, new_col)
        if new_pos not in existing_positions:
            existing_positions.add(new_pos)
            add_rows[i] = new_row
            add_cols[i] = new_col
            add_vals[i] = np.random.choice(orig_vals)
            i += 1
    
    # Combine with existing points
    result_rows = np.concatenate([base_rows, add_rows[:i]])
    result_cols = np.concatenate([base_cols, add_cols[:i]])
    result_vals = np.concatenate([base_vals, add_vals[:i]])
    
    return result_rows, result_cols, result_vals

def scale_sparse_matrix_nearest(original_matrix: sp.csr_matrix, new_size: int, match_nnz: bool = True) -> sp.csr_matrix:
    """
    Scale a sparse matrix with nearest-neighbor interpolation while maintaining the sparsity pattern.
    
    Parameters:
    -----------
    original_matrix : scipy.sparse.spmatrix
        Input sparse matrix to be scaled
    new_size : int
        New size for the matrix (will be scaled to new_size x new_size)
    output_path : str
        Output path for saving the scaled matrix as .mtx file
    match_nnz: bool
        To decide match the exact number of nonzeros or not, adjust to exactly match target_nnz if needed
        
    Returns:
    --------
    scipy.sparse.csr_matrix
        Scaled sparse matrix with preserved value range and linearly scaled number of nonzeros
    """
    # Convert to COO format to easily access coordinates and values
    original_matrix = sp.coo_matrix(original_matrix)
    
    # Get original dimensions and extract coordinates and values
    original_size = max(original_matrix.shape)
    orig_rows = original_matrix.row
    orig_cols = original_matrix.col
    orig_vals = original_matrix.data
    
    # Calculate scaling factor and target number of non-zeros
    scale_factor = new_size / original_size
    orig_nnz = len(orig_vals)
    target_nnz = int(orig_nnz * scale_factor)
    
    # Define neighborhood size for generating new points
    neighborhood_size = int(scale_factor)
    half_size = max(1, neighborhood_size // 2)
    
    # Base point selection based on scaling direction
    if scale_factor < 1:  # Downscaling
        indices_to_keep = np.random.choice(orig_nnz, size=target_nnz, replace=False)
        base_rows = orig_rows[indices_to_keep]
        base_cols = orig_cols[indices_to_keep]
        base_vals = orig_vals[indices_to_keep]
    else:  # Upscaling
        base_rows = orig_rows
        base_cols = orig_cols
        base_vals = orig_vals
    
    # Apply nearest-neighbor interpolation for all base coordinates
    scaled_rows = np.minimum(np.floor(base_rows * scale_factor).astype(int), new_size - 1)
    scaled_cols = np.minimum(np.floor(base_cols * scale_factor).astype(int), new_size - 1)
    
    # Default case: use scaled original points
    final_rows = scaled_rows
    final_cols = scaled_cols
    final_vals = base_vals
    
    # If upscaling, add new points to maintain density
    if scale_factor > 1:
        base_count = len(base_vals)
        additional_needed = target_nnz - base_count
        
        if additional_needed > 0:
            # Generate additional points using helper function
            additional_rows, additional_cols, additional_vals = generate_additional_points(
                scaled_rows, scaled_cols, additional_needed, half_size, new_size, orig_vals
            )
            
            # Combine base and additional points
            final_rows = np.concatenate([scaled_rows, additional_rows])
            final_cols = np.concatenate([scaled_cols, additional_cols])
            final_vals = np.concatenate([base_vals, additional_vals])
    
    # Handle collisions using helper function
    result_rows, result_cols, result_vals = handle_collisions(final_rows, final_cols, final_vals, new_size)
    
    # Adjust number of non-zeros if needed
    current_nnz = len(result_rows)
    if match_nnz:
        if current_nnz > target_nnz:
            # Too many non-zeros, randomly remove some
            keep_indices = np.random.choice(current_nnz, target_nnz, replace=False)
            result_rows = result_rows[keep_indices]
            result_cols = result_cols[keep_indices]
            result_vals = result_vals[keep_indices]
        elif current_nnz < target_nnz:
            # Too few non-zeros, add more using helper function
            additional_needed = target_nnz - current_nnz
            result_rows, result_cols, result_vals = add_points(
                result_rows, result_cols, result_vals, additional_needed, half_size, new_size, orig_vals
            )
    
    # Create the final sparse matrix and save it
    scaled_matrix = sp.csr_matrix((result_vals, (result_rows, result_cols)), shape=(new_size, new_size))
    
    return scaled_matrix