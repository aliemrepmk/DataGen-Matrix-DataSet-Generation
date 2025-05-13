import numpy as np
import scipy.sparse as sp
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def resize_sparse_fft(sparse_mat: sp.spmatrix, new_shape: tuple[int, int], eps: float = 1e-3) -> sp.csr_matrix:
    """
    Resize a 2D sparse matrix via Fourier zero-pad/crop **with correct fftshift**.
    Converts to dense only for FFT, then returns sparse CSR.
    """
    # Original dims and target dims
    M, N = sparse_mat.shape
    P, Q = new_shape

    # 1) Dense for FFT
    A = sparse_mat.toarray()

    # 2) Forward FFT + shift
    F = fft2(A)
    F_shift = fftshift(F)

    # 3) Allocate padded/cropped spectrum (shifted)
    F2_shift = np.zeros((P, Q), dtype=complex)

    # 4) Determine low-frequency block sizes and offsets
    low_rows = min(M, P)
    low_cols = min(N, Q)
    r_old = (M - low_rows) // 2
    c_old = (N - low_cols) // 2
    r_new = (P - low_rows) // 2
    c_new = (Q - low_cols) // 2

    # 5) Copy central block of shifted spectrum
    F2_shift[r_new:r_new + low_rows, c_new:c_new + low_cols] = \
        F_shift[r_old:r_old + low_rows, c_old:c_old + low_cols]

    # 6) Inverse shift + inverse FFT
    F2 = ifftshift(F2_shift)
    A2 = np.real(ifft2(F2))

    # 7) Threshold and convert back to sparse
    A2[np.abs(A2) < eps] = 0
    return sp.csr_matrix(A2)

def scale_sparse_matrix_fourier(original_matrix: sp.csr_matrix, new_size: int) -> sp.csr_matrix:
    """
    Scale a sparse matrix using Fourier transform-based resizing to a square shape.
    
    Parameters:
    -----------
    original_matrix : scipy.sparse.csr_matrix
        Input sparse matrix to be scaled
    new_size : int
        New size for both rows and columns (will create a square matrix)
        
    Returns:
    --------
    scipy.sparse.csr_matrix
        Scaled sparse matrix of shape (new_size, new_size)
    """
    # Apply Fourier transform-based resizing to create a square matrix
    scaled_matrix = resize_sparse_fft(original_matrix, (new_size, new_size))
    
    return scaled_matrix
