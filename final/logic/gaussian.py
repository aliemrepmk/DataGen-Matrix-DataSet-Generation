import numpy as np
import scipy.sparse as sp

def gaussian_blur_sparse(matrix, sigma=1.0, epsilon=1e-10):
    # Check if the input is a sparse matrix
    if not sp.isspmatrix(matrix):
        raise ValueError("Input must be a scipy sparse matrix.")
    
    size = int(3 * sigma)  # Determine the kernel size (3*sigma is a typical choice for Gaussian spread)

    # Generate 1D Gaussian kernel values over the window [-size, size]
    x = np.arange(-size, size + 1)
    gaussian_kernel = np.exp(-x**2 / (2 * sigma**2))
    gaussian_kernel /= gaussian_kernel.sum()  # Normalize the kernel so that sum of weights = 1

    result = {}  # Dictionary to accumulate blurred values

    # Extract non-zero row and column indices and corresponding non-zero data values
    rows, cols = matrix.nonzero()
    data = matrix.data

    # Iterate over each non-zero element in the sparse matrix
    for idx in range(len(rows)):
        r, c = rows[idx], cols[idx]  # row, column of current non-zero
        v = data[idx]                # value of current non-zero
        
        # Apply Gaussian weights over the local neighborhood
        for dy in range(-size, size + 1):  # vertical shift
            for dx in range(-size, size + 1):  # horizontal shift
                weight = gaussian_kernel[dy + size] * gaussian_kernel[dx + size]  # combined weight
                new_r, new_c = r + dy, c + dx  # new neighbor position
                
                # Check if new position is within matrix bounds
                if 0 <= new_r < matrix.shape[0] and 0 <= new_c < matrix.shape[1]:
                    key = (new_r, new_c)  # dictionary key as (row, column)
                    result[key] = result.get(key, 0.0) + v * weight  # Accumulate weighted value into the result dictionary

    # Convert dictionary keys (coordinates) and values to numpy arrays
    coords = np.array(list(result.keys()))  # array of [row, col] pairs
    values = np.array(list(result.values()))  # array of corresponding blurred values

    # Filter out very small values to maintain sparsity
    mask = np.abs(values) > epsilon
    coords = coords[mask]
    values = values[mask]

    # Create a sparse COO matrix from the filtered coordinates and values
    blurred = sp.coo_matrix((values, (coords[:,0], coords[:,1])), shape=matrix.shape)

    # Convert the result to CSR format for further usage
    return blurred.tocsr()

def downsample_sparse(matrix):
    if not sp.isspmatrix(matrix):
        raise ValueError("Input must be a scipy sparse matrix.")

    matrix = matrix.tocoo()  # Convert to COO for easy manipulation
    mask = (matrix.row % 2 == 0) & (matrix.col % 2 == 0)  # Keep only even rows and columns
    new_rows = matrix.row[mask] // 2
    new_cols = matrix.col[mask] // 2
    new_data = matrix.data[mask]

    # Use ceiling division to correctly handle odd dimensions
    new_shape = ((matrix.shape[0] + 1) // 2, (matrix.shape[1] + 1) // 2)

    # Build downsampled sparse matrix
    downsampled = sp.coo_matrix((new_data, (new_rows, new_cols)), shape=new_shape)
    return downsampled.tocsr()

def scale_sparse_matrix_gaussian(original_matrix: sp.csr_matrix, new_size: int, sigma: float = 1.0) -> sp.csr_matrix:
    if not sp.isspmatrix(original_matrix):
        raise ValueError("Input must be a scipy sparse matrix.")

    current = original_matrix  # Start with the original matrix

    # Continue blurring and downsampling until matrix is close to new_size
    while current.shape[0] > new_size * 2 or current.shape[1] > new_size * 2:
        blurred = gaussian_blur_sparse(current, sigma=sigma)  # Blur the matrix
        current = downsample_sparse(blurred)  # Downsample the blurred matrix

    # Final blur and downsample to reach exactly new_size x new_size
    blurred = gaussian_blur_sparse(current, sigma=sigma)
    final_matrix = downsample_sparse(blurred)

    # If we're still not at the target size, do one more downsample
    if final_matrix.shape[0] > new_size or final_matrix.shape[1] > new_size:
        final_matrix = downsample_sparse(final_matrix)

    return final_matrix