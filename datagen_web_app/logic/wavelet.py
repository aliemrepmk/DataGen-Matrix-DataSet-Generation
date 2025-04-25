import numpy as np
import scipy.sparse as sp
from scipy.ndimage import zoom
import pywt

def round_to_multiple(n: int, multiple: int) -> int:
    """Round a number to the nearest multiple."""
    return ((n + multiple - 1) // multiple) * multiple

def perturb_details(coeff: np.ndarray) -> np.ndarray:
    """Add random perturbation to detail coefficients."""
    if coeff.size == 0:
        return coeff
    noise = np.random.normal(0, np.std(coeff) * 0.1, coeff.shape)
    return coeff + noise

def scale_sparse_matrix_wavelet(original_matrix: sp.csr_matrix, new_rows: int, new_cols: int, wavelet_type: str = 'db4', block_size: int = 2) -> sp.csr_matrix:
    """
    Generate a new matrix using wavelet transform and reconstruction.
    Processes the matrix in blocks to handle large sparse matrices efficiently.
    
    Parameters:
    -----------
    original_matrix : scipy.sparse.csr_matrix
        Input sparse matrix to be scaled
    new_rows : int
        Number of rows in the output matrix
    new_cols : int
        Number of columns in the output matrix
    wavelet_type : str
        Type of wavelet to use (e.g., 'db4', 'sym4', 'coif3')
    block_size : int
        Size of the blocks to process (must be even)
        
    Returns:
    --------
    scipy.sparse.csr_matrix
        Scaled sparse matrix
    """
    # Ensure block_size is even
    if block_size % 2 != 0:
        block_size += 1
    
    # Round dimensions to nearest multiple of block_size
    orig_rows, orig_cols = original_matrix.shape
    padded_rows = round_to_multiple(orig_rows, block_size)
    padded_cols = round_to_multiple(orig_cols, block_size)
    
    # Process matrix in blocks
    block_rows = padded_rows // block_size
    block_cols = padded_cols // block_size
    
    # Initialize result matrix components
    result_data = []
    result_rows = []
    result_cols = []
    
    # Calculate scaling factors
    scale_rows = new_rows / orig_rows
    scale_cols = new_cols / orig_cols
    
    # Calculate target size for each block
    target_size = (int(block_size * scale_rows), int(block_size * scale_cols))
    
    # Process each block
    for i in range(block_rows):
        for j in range(block_cols):
            # Extract block as sparse matrix
            start_row = i * block_size
            start_col = j * block_size
            block = original_matrix[start_row:start_row + block_size, 
                                  start_col:start_col + block_size]
            
            # Skip empty blocks
            if block.nnz == 0:
                continue
                
            # Convert only non-zero elements to dense for wavelet transform
            block_dense = np.zeros((block_size, block_size))
            for row, col in zip(*block.nonzero()):
                block_dense[row, col] = block[row, col]
            
            # Apply two level wavelet transform
            coeffs = pywt.wavedec2(block_dense, wavelet_type, level=2)
            cA, (cH, cV, cD), (cH2, cV2, cD2) = coeffs
            
            # Add perturbation to detail coefficients (without resizing)
            cH = perturb_details(cH)
            cV = perturb_details(cV)
            cD = perturb_details(cD)
            cH2 = perturb_details(cH2)
            cV2 = perturb_details(cV2)
            cD2 = perturb_details(cD2)
            
            # Reconstruct block with original coefficient sizes
            new_coeffs = [cA, (cH, cV, cD), (cH2, cV2, cD2)]
            reconstructed = pywt.waverec2(new_coeffs, wavelet_type)
            
            # Now resize the reconstructed block to target size
            reconstructed = zoom(reconstructed, 
                              (target_size[0]/reconstructed.shape[0], 
                               target_size[1]/reconstructed.shape[1]), 
                              order=1)
            
            # Calculate new block position
            new_start_row = int(start_row * scale_rows)
            new_start_col = int(start_col * scale_cols)
            
            # Add non-zero elements to result (with thresholding)
            threshold = 1e-10
            for r in range(min(target_size[0], new_rows - new_start_row)):
                for c in range(min(target_size[1], new_cols - new_start_col)):
                    val = reconstructed[r, c]
                    if abs(val) > threshold:
                        row_idx = new_start_row + r
                        col_idx = new_start_col + c
                        if row_idx < new_rows and col_idx < new_cols:
                            result_rows.append(row_idx)
                            result_cols.append(col_idx)
                            result_data.append(val)
    
    return sp.csr_matrix((result_data, (result_rows, result_cols)), 
                        shape=(new_rows, new_cols))