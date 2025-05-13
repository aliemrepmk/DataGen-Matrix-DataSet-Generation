import random
import numpy as np
import scipy.sparse as sp

def lanczos_kernel(x: float, a: int = 3) -> float:
    """
    Lanczos kernel function with parameter a (typically 2 or 3).
    
    Parameters:
    -----------
    x : float
        Distance from the center point
    a : int
        Size of the kernel (larger values provide higher quality but more computation)
        
    Returns:
    --------
    float
        Kernel weight value
    """
    if x == 0:
        return 1.0
    elif abs(x) < a:
        return (a * np.sin(np.pi * x) * np.sin(np.pi * x / a)) / (np.pi * np.pi * x * x)
    else:
        return 0.0

def scale_sparse_matrix_lanczos(original_matrix: sp.csr_matrix, new_size: int, match_nnz: bool = True, a: int = 3) -> sp.csr_matrix:
    """
    Scale a sparse matrix with Lanczos resampling while maintaining sparsity throughout.
    
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
    a : int
        Lanczos kernel parameter, (higher values = higher quality)
        
    Returns:
    --------
    scipy.sparse.csr_matrix
        Scaled sparse matrix with preserved value range and linearly scaled number of nonzeros
    """
    # Convert to CSR format for efficient row slicing
    original_matrix = sp.csr_matrix(original_matrix)
    
    # Get original dimensions
    original_size = max(original_matrix.shape)
    
    # Calculate scaling factor
    scale_factor = new_size / original_size
    
    # Calculate target number of nonzeros
    orig_nnz = original_matrix.nnz
    target_nnz = int(orig_nnz * scale_factor)  # Scale ratio
    
    # Store original value range for later normalization
    orig_values = original_matrix.data
    orig_min_val = np.min(orig_values)
    orig_max_val = np.max(orig_values)
    
    # Create a dictionary for fast lookup of nonzero elements
    orig_nonzeros = {}
    rows, cols = original_matrix.nonzero()
    for i, (r, c) in enumerate(zip(rows, cols)):
        orig_nonzeros[(r, c)] = original_matrix[r, c]
    
    # Dictionary to store the new coordinates and values
    new_coords = {}
    
    # Calculate how many points to sample initially
    # We'll sample more than target_nnz because some might end up with zero values
    sample_count = min(int(target_nnz * 2.0), new_size * new_size)
    
    candidates = set()
    
    # First approach: Direct mapping from original nonzeros (50% of samples)
    direct_samples = min(len(orig_nonzeros), sample_count // 2)
    original_positions = list(orig_nonzeros.keys())
    selected_positions = random.sample(original_positions, direct_samples)
    
    for orig_row, orig_col in selected_positions:
        # Scale the position to the new matrix
        new_row = min(int(orig_row * scale_factor), new_size - 1)
        new_col = min(int(orig_col * scale_factor), new_size - 1)
        
        # Apply small random jitter to spread out the samples
        jitter = random.randint(-1, 1)
        new_row = max(0, min(new_size - 1, new_row + jitter))
        new_col = max(0, min(new_size - 1, new_col + jitter))
        
        # Each new position becomes a candidate for Lanczos resampling
        if (new_row, new_col) not in new_coords:
            candidates.add((new_row, new_col))
    
    # Second approach: Random sampling across the matrix (remaining samples)
    while len(candidates) < sample_count:
        new_row = random.randint(0, new_size - 1)
        new_col = random.randint(0, new_size - 1)
        candidates.add((new_row, new_col))
    
    # Perform Lanczos resampling for each candidate position
    for new_row, new_col in candidates:
        # Map back to original matrix coordinates
        orig_row_float = new_row / scale_factor
        orig_col_float = new_col / scale_factor
        
        # Apply Lanczos filter in 2D
        value_sum = 0.0
        weight_sum = 0.0
        
        # Iterate over the Lanczos kernel window
        for i in range(-a+1, a+1):
            for j in range(-a+1, a+1):
                # Calculate the sample point in the original matrix
                sample_row = int(np.floor(orig_row_float + i))
                sample_col = int(np.floor(orig_col_float + j))
                
                # Skip if outside the original matrix
                if (sample_row < 0 or sample_row >= original_size or 
                    sample_col < 0 or sample_col >= original_size):
                    continue
                
                # Calculate distances for kernel
                dx = orig_row_float - sample_row
                dy = orig_col_float - sample_col
                
                # Apply Lanczos kernel in 2D (separable for efficiency)
                weight_x = lanczos_kernel(dx, a)
                weight_y = lanczos_kernel(dy, a)
                weight = weight_x * weight_y
                
                # Get the value from the original matrix - use sparse lookup
                orig_val = orig_nonzeros.get((sample_row, sample_col), 0.0)
                
                # Accumulate weighted values
                value_sum += orig_val * weight
                weight_sum += weight
        
        # Normalize by the sum of weights if not zero
        if weight_sum > 0:
            final_value = value_sum / weight_sum
            
            # Only add non-zero values to our result
            if abs(final_value) > 1e-10:  # Small threshold to handle floating point errors
                new_coords[(new_row, new_col)] = final_value
    
    # Ensure the value range is preserved (optional)
    # This will scale the values to match the original range
    if len(new_coords) > 0:
        values = np.array(list(new_coords.values()))
        
        # First handle negative values if the original didn't have them
        if orig_min_val >= 0:
            # Simply clamp all negative values to zero
            for pos, val in list(new_coords.items()):
                if val < 0:
                    new_coords[pos] = 0
            
            # Get updated values after clamping
            values = np.array(list(new_coords.values()))
        
        # Now normalize to match the original range exactly
        new_min_val = np.min(values)
        new_max_val = np.max(values)
        
        range_ratio = (orig_max_val - orig_min_val) / (new_max_val - new_min_val) if new_max_val > new_min_val else 1.0
        
        # Apply the exact range mapping to each value
        for pos, val in list(new_coords.items()):
            # Map to original range
            normalized = (val - new_min_val) * range_ratio + orig_min_val
            new_coords[pos] = normalized
    
    # Ensure we have exactly target_nnz non-zeros if required
    if match_nnz:
        current_coords = list(new_coords.items())
        current_nnz = len(current_coords)
        
        if current_nnz > target_nnz:
            # Too many non-zeros, randomly remove some
            indices_to_keep = random.sample(range(current_nnz), target_nnz)
            new_coords = {current_coords[i][0]: current_coords[i][1] for i in indices_to_keep}
        elif current_nnz < target_nnz:
            # Too few non-zeros, add more using a modified approach
            
            # Neighborhood size for generating new nonzeros
            neighborhood_size = max(1, int(2 * scale_factor))
            
            # List of existing positions to use as seeds
            existing_positions = list(new_coords.keys())
            
            while len(new_coords) < target_nnz and existing_positions:
                # Pick a random existing nonzero as a seed
                base_row, base_col = random.choice(existing_positions)
                
                # Try to find a nearby empty position
                attempts = 0
                max_attempts = 10
                
                while attempts < max_attempts:
                    # Random offset within neighborhood
                    offset_row = random.randint(-neighborhood_size, neighborhood_size)
                    offset_col = random.randint(-neighborhood_size, neighborhood_size)
                    
                    new_row = max(0, min(new_size - 1, base_row + offset_row))
                    new_col = max(0, min(new_size - 1, base_col + offset_col))
                    
                    if (new_row, new_col) not in new_coords:
                        # Map back to original matrix for Lanczos sampling
                        orig_row_float = new_row / scale_factor
                        orig_col_float = new_col / scale_factor
                        
                        # Apply Lanczos filter
                        value_sum = 0.0
                        weight_sum = 0.0
                        
                        for i in range(-a+1, a+1):
                            for j in range(-a+1, a+1):
                                sample_row = int(np.floor(orig_row_float + i))
                                sample_col = int(np.floor(orig_col_float + j))
                                
                                if (sample_row < 0 or sample_row >= original_size or 
                                    sample_col < 0 or sample_col >= original_size):
                                    continue
                                
                                dx = orig_row_float - sample_row
                                dy = orig_col_float - sample_col
                                
                                weight_x = lanczos_kernel(dx, a)
                                weight_y = lanczos_kernel(dy, a)
                                weight = weight_x * weight_y
                                
                                # Use sparse lookup
                                orig_val = orig_nonzeros.get((sample_row, sample_col), 0.0)
                                
                                value_sum += orig_val * weight
                                weight_sum += weight
                        
                        if weight_sum > 0:
                            final_value = value_sum / weight_sum
                            
                            # Only add if it's a significant value
                            if abs(final_value) > 1e-10:
                                new_coords[(new_row, new_col)] = final_value
                                break
                    
                    attempts += 1
                
                # If we couldn't find a suitable position after several attempts,
                # just add a value from the original matrix
                if len(new_coords) < target_nnz and attempts >= max_attempts:
                    # Find a new random position
                    new_row = random.randint(0, new_size - 1)
                    new_col = random.randint(0, new_size - 1)
                    
                    if (new_row, new_col) not in new_coords:
                        # Use a random value from the original matrix
                        new_coords[(new_row, new_col)] = random.choice(orig_values)
    
    # Convert the dictionary to COO format
    result_rows = []
    result_cols = []
    result_vals = []
    
    for (row, col), val in new_coords.items():
        result_rows.append(row)
        result_cols.append(col)
        result_vals.append(val)
    
    # Create the final sparse matrix
    scaled_matrix = sp.csr_matrix((result_vals, (result_rows, result_cols)), shape=(new_size, new_size))
    
    return scaled_matrix