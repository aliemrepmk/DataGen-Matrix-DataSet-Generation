import random
import numpy as np
import scipy.io as sio
from scipy import sparse

def scale_sparse_matrix_nearest(original_matrix: sparse.csr_matrix, new_size: int, output_path: str, match_nnz: bool = True) -> sparse.csr_matrix:
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
    original_matrix = sparse.coo_matrix(original_matrix)
    
    # Get original dimensions
    original_size = max(original_matrix.shape)
    
    # Calculate scaling factor
    scale_factor = new_size / original_size
    
    # Get coordinates and values of nonzeros in the original matrix
    orig_rows = original_matrix.row
    orig_cols = original_matrix.col
    orig_vals = original_matrix.data
    
    # Calculate how many nonzeros we should have in the scaled matrix
    orig_nnz = len(orig_vals)
    target_nnz = int(orig_nnz * scale_factor)
    
    # Define half_size for neighborhood (needed for both upscaling and match_nnz)
    neighborhood_size = int(scale_factor)
    half_size = max(1, neighborhood_size // 2)
    
    # Determine how many original nonzeros to keep
    if scale_factor < 1:
        # Downscaling: randomly select a subset of original nonzeros
        indices_to_keep = np.random.choice(orig_nnz, size=target_nnz, replace=False)
        base_rows = orig_rows[indices_to_keep]
        base_cols = orig_cols[indices_to_keep]
        base_vals = orig_vals[indices_to_keep]
    else:
        # Upscaling: keep all original nonzeros
        base_rows = orig_rows
        base_cols = orig_cols
        base_vals = orig_vals
    
    # Apply nearest-neighbor interpolation for all base coordinates
    scaled_rows = np.minimum(np.floor(base_rows * scale_factor).astype(int), new_size - 1)
    scaled_cols = np.minimum(np.floor(base_cols * scale_factor).astype(int), new_size - 1)
    
    # Default case: no additional points
    final_rows = scaled_rows
    final_cols = scaled_cols
    final_vals = base_vals
    
    base_count = len(base_vals)
    
    # If we're upscaling, add the new nonzeros
    if scale_factor > 1:
        additional_needed = target_nnz - base_count
        if additional_needed > 0:
            # Randomly select some existing nonzeros to generate neighbors for
            indices_for_neighbors = np.random.choice(base_count, size=additional_needed, replace=True)
            
            # Pre-allocate arrays for efficiency
            additional_rows = np.zeros(additional_needed, dtype=int)
            additional_cols = np.zeros(additional_needed, dtype=int)
            additional_vals = np.zeros(additional_needed, dtype=orig_vals.dtype)

            # Generate new nonzeros near selected existing ones
            for i, idx in enumerate(indices_for_neighbors):
                base_row = scaled_rows[idx]
                base_col = scaled_cols[idx]

                # Generate random offsets within the neighborhood
                row_offset = random.randint(-half_size, half_size)
                col_offset = random.randint(-half_size, half_size)

                additional_rows[i] = min(max(0, base_row + row_offset), new_size - 1)
                additional_cols[i] = min(max(0, base_col + col_offset), new_size - 1)
                additional_vals[i] = np.random.choice(orig_vals)

            # Combine base and additional nonzeros
            final_rows = np.concatenate([scaled_rows, additional_rows])
            final_cols = np.concatenate([scaled_cols, additional_cols])
            final_vals = np.concatenate([base_vals, additional_vals])

    # Handle collisions more efficiently using a structured approach with numpy
    positions = final_rows * new_size + final_cols
    unique_positions, inverse_indices = np.unique(positions, return_inverse=True)
    
    # For each unique position, find the max value among duplicates
    result_vals = np.zeros(len(unique_positions), dtype=final_vals.dtype)
    for i, val in enumerate(final_vals):
        pos_idx = inverse_indices[i]
        result_vals[pos_idx] = max(result_vals[pos_idx], val) if result_vals[pos_idx] != 0 else val
    
    # Convert unique positions back to row, col format
    result_rows = unique_positions // new_size
    result_cols = unique_positions % new_size
    
    current_nnz = len(result_rows)
    
    # If we want to have exact (scale_factor * orig_vals) number of nonzeros
    if match_nnz:
        if current_nnz > target_nnz:
            # Too many nonzeros, randomly remove some
            keep_indices = np.random.choice(current_nnz, target_nnz, replace=False)
            result_rows = result_rows[keep_indices]
            result_cols = result_cols[keep_indices]
            result_vals = result_vals[keep_indices]
        elif current_nnz < target_nnz:
            # Too few nonzeros, add more
            additional_needed = target_nnz - current_nnz
            
            # Pre-allocate arrays for the new points
            add_rows = np.zeros(additional_needed, dtype=int)
            add_cols = np.zeros(additional_needed, dtype=int)
            add_vals = np.zeros(additional_needed, dtype=orig_vals.dtype)
            
            # Create a set of existing positions for faster lookup
            existing_positions = set(zip(result_rows, result_cols))
            
            i = 0
            while i < additional_needed:
                # Choose a random existing nonzero to add a neighbor
                idx = random.randint(0, current_nnz - 1)
                base_row = result_rows[idx]
                base_col = result_cols[idx]

                # Generate a nearby position
                row_offset = random.randint(-half_size, half_size)
                col_offset = random.randint(-half_size, half_size)

                new_row = min(max(0, base_row + row_offset), new_size - 1)
                new_col = min(max(0, base_col + col_offset), new_size - 1)
                
                # Check if position is already occupied
                new_pos = (new_row, new_col)
                if new_pos not in existing_positions:
                    existing_positions.add(new_pos)
                    add_rows[i] = new_row
                    add_cols[i] = new_col
                    add_vals[i] = np.random.choice(orig_vals)
                    i += 1
            
            # Combine with existing points
            result_rows = np.concatenate([result_rows, add_rows[:i]])
            result_cols = np.concatenate([result_cols, add_cols[:i]])
            result_vals = np.concatenate([result_vals, add_vals[:i]])

    # Create the final sparse matrix and save it to output_path
    scaled_matrix = sparse.csr_matrix((result_vals, (result_rows, result_cols)), shape=(new_size, new_size))
    sio.mmwrite(output_path, scaled_matrix)

    return scaled_matrix