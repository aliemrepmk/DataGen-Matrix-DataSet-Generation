import random
import numpy as np
import scipy.sparse as sp

def bilinear_interpolation(csr_matrix: sp.csr_matrix, row: int, col: int, scale_factor: float, original_size: int):
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
    orig_row_float = row / scale_factor  # Convert target row index to corresponding position in original matrix
    orig_col_float = col / scale_factor  # Convert target column index to corresponding position in original matrix

    # Identify four surrounding neighbors in original matrix    
    row_low = int(np.floor(orig_row_float))  # Get lower row index by flooring the fractional row position
    row_high = min(row_low + 1, original_size - 1)  # Get upper row index, ensuring we don't go beyond matrix boundaries
    col_low = int(np.floor(orig_col_float))  # Get left column index by flooring the fractional column position
    col_high = min(col_low + 1, original_size - 1)  # Get right column index, ensuring we don't go beyond matrix boundaries

    # Compute weights based on relative position within the cell
    w_row = orig_row_float - row_low  # Calculate vertical interpolation factor (0-1)
    w_col = orig_col_float - col_low  # Calculate horizontal interpolation factor (0-1)

    # Retrieve the four corner values from the original sparse matrix
    val_ll = csr_matrix[row_low, col_low]   # lower-left value from original matrix
    val_lh = csr_matrix[row_low, col_high]  # lower-right value from original matrix
    val_hl = csr_matrix[row_high, col_low]  # upper-left value from original matrix
    val_hh = csr_matrix[row_high, col_high] # upper-right value from original matrix

    # Linearly interpolate between columns and rows
    top = (1 - w_col) * val_ll + w_col * val_lh  # Horizontally interpolate along the top edge based on column weight
    bottom = (1 - w_col) * val_hl + w_col * val_hh  # Horizontally interpolate along the bottom edge based on column weight
    return (1 - w_row) * top + w_row * bottom  # Vertically interpolate between top and bottom results based on row weight

def scale_sparse_matrix_bilinear(original_matrix: sp.csr_matrix, new_size: int, match_nnz = True) -> sp.csr_matrix:
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
    original_csr = sp.csr_matrix(original_matrix)
    
    # Get original dimensions and values
    original_size = max(original_matrix.shape)  # Get the maximum dimension
    orig_nnz = original_matrix.nnz  # Get the number of non-zero elements in the original matrix
    
    # Calculate scaling factor and target nonzeros
    scale_factor = new_size / original_size  # Calculate how much larger/smaller the new matrix will be
    target_nnz = int(orig_nnz * scale_factor)  # Scale the number of non-zeros proportionally to maintain similar density
    
    # Create a dictionary to store new coordinates and values
    new_coords = {}  # Dictionary to store (row,col)
    
    # Calculate how many points to sample initially
    sample_count = int(target_nnz * 1.5)  # Sample 50% more points than needed to account for zeros after interpolation
    
    # Generate random coordinates in the new matrix
    candidates = set()  # Use a set to avoid duplicate coordinates for efficiency
    while len(candidates) < sample_count:  # Continue until we have enough unique coordinates
        new_row = random.randint(0, new_size - 1)  # Generate random row index within the new matrix bounds
        new_col = random.randint(0, new_size - 1)  # Generate random column index within the new matrix bounds
        candidates.add((new_row, new_col))  # Add the coordinate pair to our set of candidates
    
    # For each sampled coordinate, perform bilinear interpolation
    for new_row, new_col in candidates:  # Iterate through each candidate coordinate
        value = bilinear_interpolation(original_csr, new_row, new_col, scale_factor, original_size)  # Calculate interpolated value
        # Only add non-zero values to our result
        if abs(value) > 1e-10:  # Check if value is non-zero
            new_coords[(new_row, new_col)] = value  # Store the non-zero value in the dictionary
    
    # Ensure to have exactly target_nnz non-zeros if required
    if match_nnz:  # If we need to match the target number of non-zeros exactly
        current_coords = list(new_coords.items())  # Convert dictionary items to list for random sampling
        current_nnz = len(current_coords)  # Count how many non-zeros we currently have
        
        if current_nnz > target_nnz:  # If we have too many non-zeros
            # Too many non-zeros, randomly remove some
            new_coords = dict(random.sample(current_coords, target_nnz))  # Randomly select exact number needed
        elif current_nnz < target_nnz:  # If we have too few non-zeros
            # Too few non-zeros, add more
            while len(new_coords) < target_nnz:  # Continue until we reach the target
                new_row = random.randint(0, new_size - 1)  # Generate random row index within the new matrix bounds
                new_col = random.randint(0, new_size - 1)  # Generate random column index within the new matrix bounds
                
                if (new_row, new_col) not in new_coords:  # Check if this coordinate is already used
                    value = bilinear_interpolation(original_csr, new_row, new_col, scale_factor, original_size)  # Calculate interpolated value
                    if abs(value) > 1e-10:  # Check if value is non-zero
                        new_coords[(new_row, new_col)] = value  # Store the non-zero value in the dictionary
    
    # Convert the dictionary to COO format
    result_rows = []  # List to store row indices for COO format
    result_cols = []  # List to store column indices for COO format
    result_vals = []  # List to store values for COO format
    
    for (row, col), val in new_coords.items():  # Iterate through dictionary of non-zeros
        result_rows.append(row)  # Add row index to list
        result_cols.append(col)  # Add column index to list
        result_vals.append(val)  # Add value to list
    
    # Create the final sparse matrix and save it to output_path
    scaled_matrix = sp.csr_matrix((result_vals, (result_rows, result_cols)), shape=(new_size, new_size))
    
    return scaled_matrix  # Return the scaled matrix