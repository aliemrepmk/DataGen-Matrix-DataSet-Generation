import random
import numpy as np
import scipy.io as sio
from scipy import sparse

def scale_sparse_matrix_gaussian(original_matrix: sparse.csr_matrix, new_size: int, output_path: str, match_nnz: bool = True, sigma: float = 1.0) -> sparse.csr_matrix:
    """
    Scale a sparse matrix using Gaussian pyramid approach for downscaling.
    
    Parameters:
    -----------
    original_matrix : scipy.sparse.spmatrix
        Input sparse matrix to be scaled
    new_size : int
        New size for the matrix (will be scaled to new_size x new_size)
    output_path : str
        Output path for saving the scaled matrix as .mtx file
    sigma : float
        Standard deviation for the Gaussian kernel
    match_nnz : bool
        Whether to match the exact number of nonzeros based on linear scaling factor
        
    Returns:
    --------
    scipy.sparse.csr_matrix
        Downscaled sparse matrix using Gaussian pyramid approach
    """
    # Convert to COO format to access coordinates and values
    original_matrix = sparse.coo_matrix(original_matrix)
    
    # Get original dimensions
    original_size = max(original_matrix.shape)
    
    # Calculate scaling factor
    scale_factor = new_size / original_size
    
    # Only proceed with Gaussian pyramids for downscaling
    if scale_factor >= 1:
        print("Warning: Gaussian pyramid is designed for downscaling. Using nearest neighbor for upscaling.")
        # Call the nearest neighbor method for upscaling (you can import it from your existing code)
        from nearest import scale_sparse_matrix_nearest
        return scale_sparse_matrix_nearest(original_matrix, new_size, output_path, match_nnz)
    
    # Get coordinates and values of nonzeros
    orig_rows = original_matrix.row
    orig_cols = original_matrix.col
    orig_vals = original_matrix.data
    
    # Calculate target number of nonzeros
    orig_nnz = len(orig_vals)
    target_nnz = int(orig_nnz * scale_factor)
    
    # Step 1: Create a sparse COO representation for efficient lookup
    sparse_dict = {(r, c): v for r, c, v in zip(orig_rows, orig_cols, orig_vals)}
    
    # Step 2: Calculate the Gaussian kernel size based on scaling factor
    # Kernel size should be odd and scaled according to downsampling factor
    kernel_size = max(3, int(1 / scale_factor * 2 + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure kernel size is odd
    
    # Step 3: Create Gaussian kernel
    kernel_radius = kernel_size // 2
    gaussian_kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - kernel_radius
            y = j - kernel_radius
            gaussian_kernel[i, j] = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
    
    # Normalize the kernel
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
    
    # Step 4: Determine the grid points in the new matrix
    # For each point in the output, we'll compute a weighted average from the input
    result_rows = []
    result_cols = []
    result_vals = []
    
    # Create a set to track positions we've already filled
    filled_positions = set()
    
    # Generate the initial set of points by applying Gaussian filtering
    for new_row in range(new_size):
        for new_col in range(new_size):
            # Map back to original coordinates
            orig_center_row = int(new_row / scale_factor)
            orig_center_col = int(new_col / scale_factor)
            
            # Skip if we've already processed this position
            if (new_row, new_col) in filled_positions:
                continue
            
            # Apply Gaussian filtering
            weighted_sum = 0
            total_weight = 0
            has_nonzero_neighbors = False
            
            for k_row in range(-kernel_radius, kernel_radius + 1):
                for k_col in range(-kernel_radius, kernel_radius + 1):
                    orig_row = orig_center_row + k_row
                    orig_col = orig_center_col + k_col
                    
                    # Skip if outside original matrix bounds
                    if (orig_row < 0 or orig_row >= original_size or 
                        orig_col < 0 or orig_col >= original_size):
                        continue
                    
                    # Get value at this position (0 if no nonzero)
                    value = sparse_dict.get((orig_row, orig_col), 0)
                    
                    if value != 0:
                        has_nonzero_neighbors = True
                        weight = gaussian_kernel[k_row + kernel_radius, k_col + kernel_radius]
                        weighted_sum += value * weight
                        total_weight += weight
            
            # Only add a nonzero if there were nonzero neighbors
            if has_nonzero_neighbors and total_weight > 0:
                # Compute weighted average
                new_val = weighted_sum / total_weight
                
                # Add to result
                result_rows.append(new_row)
                result_cols.append(new_col)
                result_vals.append(new_val)
                filled_positions.add((new_row, new_col))
    
    # Step 5: Adjust the number of nonzeros if needed
    current_nnz = len(result_rows)
    if match_nnz:
        if current_nnz > target_nnz:
            # Too many nonzeros, randomly remove some
            indices = np.random.choice(current_nnz, size=target_nnz, replace=False)
            result_rows = [result_rows[i] for i in indices]
            result_cols = [result_cols[i] for i in indices]
            result_vals = [result_vals[i] for i in indices]
        elif current_nnz < target_nnz:
            # Too few nonzeros, add more by sampling from existing nonzeros
            # and adding small random perturbations
            additional_needed = target_nnz - current_nnz
            for _ in range(additional_needed):
                # Select a random existing nonzero
                if current_nnz > 0:
                    idx = random.randint(0, current_nnz - 1)
                    base_row = result_rows[idx]
                    base_col = result_cols[idx]
                    base_val = result_vals[idx]
                else:
                    # If there are no existing nonzeros, create some
                    base_row = random.randint(0, new_size - 1)
                    base_col = random.randint(0, new_size - 1)
                    base_val = random.choice(orig_vals) if len(orig_vals) > 0 else 1.0
                
                # Try to find an unoccupied position nearby
                max_attempts = 10
                for attempt in range(max_attempts):
                    # Small perturbation
                    offset = random.randint(1, max(2, int(1/scale_factor)))
                    direction = random.randint(0, 3)
                    
                    new_row = base_row
                    new_col = base_col
                    
                    if direction == 0:
                        new_row = min(new_size - 1, base_row + offset)
                    elif direction == 1:
                        new_row = max(0, base_row - offset)
                    elif direction == 2:
                        new_col = min(new_size - 1, base_col + offset)
                    else:
                        new_col = max(0, base_col - offset)
                    
                    # Check if position is not occupied
                    if (new_row, new_col) not in filled_positions:
                        # Small value perturbation
                        new_val = base_val * (0.9 + 0.2 * random.random())
                        
                        result_rows.append(new_row)
                        result_cols.append(new_col)
                        result_vals.append(new_val)
                        filled_positions.add((new_row, new_col))
                        break
    
    # Create the final sparse matrix and save it to output_path
    scaled_matrix = sparse.csr_matrix((result_vals, (result_rows, result_cols)), shape=(new_size, new_size))
    sio.mmwrite(output_path, scaled_matrix)
    
    return scaled_matrix