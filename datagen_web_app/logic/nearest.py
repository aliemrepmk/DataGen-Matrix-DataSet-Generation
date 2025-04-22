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
    original_size = max(original_matrix.shape)  # Get the maximum dimension
    
    # Calculate scaling factor
    scale_factor = new_size / original_size  # Calculate how much larger/smaller the new matrix will be
    
    # Get coordinates and values of nonzeros in the original matrix
    orig_rows = original_matrix.row  # Extract row indices of all non-zero elements from the COO matrix
    orig_cols = original_matrix.col  # Extract column indices of all non-zero elements from the COO matrix
    orig_vals = original_matrix.data  # Extract values of all non-zero elements from the COO matrix
    
    # Calculate how many nonzeros we should have in the scaled matrix
    orig_nnz = len(orig_vals)   # Count the number of non-zero elements in the original matrix
    target_nnz = int(orig_nnz * scale_factor)  # Scale the number of non-zeros proportionally to maintain similar density
    
    # Define half_size for neighborhood (needed for both upscaling and match_nnz)
    neighborhood_size = int(scale_factor)  # Use scaling factor to determine neighborhood size for generating new points
    half_size = max(1, neighborhood_size // 2)  # Calculate half the neighborhood size, ensuring it's at least 1
    
    # Determine how many original nonzeros to keep
    if scale_factor < 1:  # If we're downscaling (new matrix is smaller than original)
        # Downscaling: randomly select a subset of original nonzeros
        indices_to_keep = np.random.choice(orig_nnz, size=target_nnz, replace=False)  # Randomly select which original points to keep
        base_rows = orig_rows[indices_to_keep]  # Get the row indices of selected points
        base_cols = orig_cols[indices_to_keep]  # Get the column indices of selected points
        base_vals = orig_vals[indices_to_keep]  # Get the values of selected points
    else:  # If we're upscaling (new matrix is larger than original)
        # Upscaling: keep all original nonzeros
        base_rows = orig_rows  # Keep all original row indices
        base_cols = orig_cols  # Keep all original column indices
        base_vals = orig_vals  # Keep all original values
    
    # Apply nearest-neighbor interpolation for all base coordinates
    scaled_rows = np.minimum(np.floor(base_rows * scale_factor).astype(int), new_size - 1)  # Scale row indices, floor to get nearest neighbor, ensure within bounds
    scaled_cols = np.minimum(np.floor(base_cols * scale_factor).astype(int), new_size - 1)  # Scale column indices, floor to get nearest neighbor, ensure within bounds
    
    # Default case: no additional points
    final_rows = scaled_rows  # Initialize final row indices with scaled original points
    final_cols = scaled_cols  # Initialize final column indices with scaled original points
    final_vals = base_vals  # Initialize final values with original values
    
    base_count = len(base_vals)  # Count how many base points we have after the initial scaling
    
    # If we're upscaling, add the new nonzeros
    if scale_factor > 1:  # When upscaling, we need to add more points to maintain proper density
        additional_needed = target_nnz - base_count  # Calculate how many additional points we need
        if additional_needed > 0:  # Only proceed if we actually need more points
            # Randomly select some existing nonzeros to generate neighbors for
            indices_for_neighbors = np.random.choice(base_count, size=additional_needed, replace=True)  # Sample with replacement to allow multiple neighbors per point
            
            # Pre-allocate arrays for efficiency
            additional_rows = np.zeros(additional_needed, dtype=int)  # Create array to store row indices of new points
            additional_cols = np.zeros(additional_needed, dtype=int)  # Create array to store column indices of new points
            additional_vals = np.zeros(additional_needed, dtype=orig_vals.dtype)  # Create array to store values of new points, matching original data type

            # Generate new nonzeros near selected existing ones
            for i, idx in enumerate(indices_for_neighbors):  # For each point we need to add
                base_row = scaled_rows[idx]  # Get the row of the selected existing point
                base_col = scaled_cols[idx]  # Get the column of the selected existing point

                # Generate random offsets within the neighborhood
                row_offset = random.randint(-half_size, half_size)  # Random vertical offset within neighborhood
                col_offset = random.randint(-half_size, half_size)  # Random horizontal offset within neighborhood

                additional_rows[i] = min(max(0, base_row + row_offset), new_size - 1)  # Apply offset to row, ensuring it stays within matrix bounds
                additional_cols[i] = min(max(0, base_col + col_offset), new_size - 1)  # Apply offset to column, ensuring it stays within matrix bounds
                additional_vals[i] = np.random.choice(orig_vals)  # Randomly select a value from the original matrix for this new point

            # Combine base and additional nonzeros
            final_rows = np.concatenate([scaled_rows, additional_rows])  # Combine original and new row indices
            final_cols = np.concatenate([scaled_cols, additional_cols])  # Combine original and new column indices
            final_vals = np.concatenate([base_vals, additional_vals])  # Combine original and new values

    # Handle collisions more efficiently using a structured approach with numpy
    positions = final_rows * new_size + final_cols  # Convert 2D indices to unique 1D positions using row-major flattening
    unique_positions, inverse_indices = np.unique(positions, return_inverse=True)  # Find unique positions and mapping from original to unique
    
    # For each unique position, find the max value among duplicates
    result_vals = np.zeros(len(unique_positions), dtype=final_vals.dtype)  # Initialize array for values at unique positions
    for i, val in enumerate(final_vals):  # For each value in our combined array
        pos_idx = inverse_indices[i]  # Get the index of its unique position
        result_vals[pos_idx] = max(result_vals[pos_idx], val) if result_vals[pos_idx] != 0 else val  # Keep the maximum value for each position
    
    # Convert unique positions back to row, col format
    result_rows = unique_positions // new_size  # Convert flat indices back to row indices
    result_cols = unique_positions % new_size  # Convert flat indices back to column indices
    
    current_nnz = len(result_rows)  # Count how many non-zeros we have after handling collisions
    
    # If we want to have exact (scale_factor * orig_vals) number of nonzeros
    if match_nnz:  # If we need to match the target number of non-zeros exactly
        if current_nnz > target_nnz:  # If we have too many non-zeros
            # Too many nonzeros, randomly remove some
            keep_indices = np.random.choice(current_nnz, target_nnz, replace=False)  # Randomly select which points to keep
            result_rows = result_rows[keep_indices]  # Keep only selected row indices
            result_cols = result_cols[keep_indices]  # Keep only selected column indices
            result_vals = result_vals[keep_indices]  # Keep only selected values
        elif current_nnz < target_nnz:  # If we have too few non-zeros
            # Too few nonzeros, add more
            additional_needed = target_nnz - current_nnz  # Calculate how many more points we need
            
            # Pre-allocate arrays for the new points
            add_rows = np.zeros(additional_needed, dtype=int)  # Array for row indices of additional points
            add_cols = np.zeros(additional_needed, dtype=int)  # Array for column indices of additional points
            add_vals = np.zeros(additional_needed, dtype=orig_vals.dtype)  # Array for values of additional points, matching original data type
            
            # Create a set of existing positions for faster lookup
            existing_positions = set(zip(result_rows, result_cols))
            
            i = 0  # Initialize counter for successfully added points
            while i < additional_needed:  # Continue until we've added enough points
                # Choose a random existing nonzero to add a neighbor
                idx = random.randint(0, current_nnz - 1)  # Randomly select an existing point
                base_row = result_rows[idx]  # Get its row index
                base_col = result_cols[idx]  # Get its column index

                # Generate a nearby position
                row_offset = random.randint(-half_size, half_size)  # Random vertical offset within neighborhood
                col_offset = random.randint(-half_size, half_size)  # Random horizontal offset within neighborhood

                new_row = min(max(0, base_row + row_offset), new_size - 1)  # Apply offset to row, ensuring it stays within matrix bounds
                new_col = min(max(0, base_col + col_offset), new_size - 1)  # Apply offset to column, ensuring it stays within matrix bounds
                
                # Check if position is already occupied
                new_pos = (new_row, new_col)  # Create a tuple representing the position
                if new_pos not in existing_positions:  # Only add the point if its position is not already occupied
                    existing_positions.add(new_pos)  # Add the new position to our set of occupied positions
                    add_rows[i] = new_row  # Store the new row index
                    add_cols[i] = new_col  # Store the new column index
                    add_vals[i] = np.random.choice(orig_vals)  # Randomly select a value from the original matrix
                    i += 1  # Increment our counter of successfully added points
            
            # Combine with existing points
            result_rows = np.concatenate([result_rows, add_rows[:i]])  # Combine original and additional row indices
            result_cols = np.concatenate([result_cols, add_cols[:i]])  # Combine original and additional column indices
            result_vals = np.concatenate([result_vals, add_vals[:i]])  # Combine original and additional values

    # Create the final sparse matrix and save it to output_path
    scaled_matrix = sparse.csr_matrix((result_vals, (result_rows, result_cols)), shape=(new_size, new_size))
    sio.mmwrite(output_path, scaled_matrix)

    return scaled_matrix  # Return the scaled matrix