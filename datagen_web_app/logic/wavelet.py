import numpy as np
import scipy.sparse as sp
import pywt
from skimage.transform import resize

def get_scaled_shape(matrix, scale_rows, scale_cols):
    h, w = matrix.shape
    return (int(round(h * scale_rows)), int(round(w * scale_cols)))

def resize_exact(matrix, scale_rows, scale_cols):
    target_shape = get_scaled_shape(matrix, scale_rows, scale_cols)
    return resize(
        matrix,
        output_shape=target_shape,
        order=0,  # Nearest-neighbor
        mode='constant',  # Fill outside with zeros
        cval=0,
        anti_aliasing=False,
        preserve_range=True
    )

def perturb_details(coeff: np.ndarray) -> np.ndarray:
    """Add random perturbation to detail coefficients."""
    if coeff.size == 0:
        return coeff
    noise = np.random.normal(0, np.std(coeff) * 0.1, coeff.shape)
    return coeff + noise

def scale_sparse_matrix_wavelet(original_matrix: sp.csr_matrix, new_rows: int, new_cols: int, wavelet_type: str = 'db1', level: int = 2) -> sp.csr_matrix:
    block_size = (2 ** level)

    rows, cols = original_matrix.nonzero()
    values = original_matrix.data

    blocks = {}
    for i, (row, col) in enumerate(zip(rows, cols)):
        block_row = row // block_size
        block_col = col // block_size
        block_key = (block_row, block_col)
        if block_key not in blocks:
            blocks[block_key] = {'positions': [], 'values': []}
        rel_row = row % block_size
        rel_col = col % block_size
        blocks[block_key]['positions'].append((rel_row, rel_col))
        blocks[block_key]['values'].append(values[i])

    orig_rows, orig_cols = original_matrix.shape
    scale_rows = new_rows / orig_rows
    scale_cols = new_cols / orig_cols

    result_data = []
    result_rows = []
    result_cols = []

    for (block_row, block_col), block_data in blocks.items():
        block_dense = np.zeros((block_size, block_size))
        for (rel_row, rel_col), value in zip(block_data['positions'], block_data['values']):
            block_dense[rel_row, rel_col] = value

        # Pad if block too small
        required_size = pywt.Wavelet(wavelet_type).dec_len * (2 ** level)
        pad_rows = max(0, required_size - block_dense.shape[0])
        pad_cols = max(0, required_size - block_dense.shape[1])
        if pad_rows > 0 or pad_cols > 0:
            block_dense = np.pad(block_dense, ((0, pad_rows), (0, pad_cols)), mode='constant')

        try:
            coeffs = pywt.wavedec2(block_dense, wavelet_type, level=level)
        except ValueError:
            coeffs = pywt.wavedec2(block_dense, 'db1', level=level)

        if level == 1:
            cA, (cH, cV, cD) = coeffs
            cA = resize_exact(cA, scale_rows, scale_cols)
            cH = perturb_details(resize_exact(cH, scale_rows, scale_cols))
            cV = perturb_details(resize_exact(cV, scale_rows, scale_cols))
            cD = perturb_details(resize_exact(cD, scale_rows, scale_cols))
            new_coeffs = [cA, (cH, cV, cD)]

        elif level == 2:
            cA, (cH1, cV1, cD1), (cH2, cV2, cD2) = coeffs
            cA = resize_exact(cA, scale_rows, scale_cols)
            cH1 = perturb_details(resize_exact(cH1, scale_rows, scale_cols))
            cV1 = perturb_details(resize_exact(cV1, scale_rows, scale_cols))
            cD1 = perturb_details(resize_exact(cD1, scale_rows, scale_cols))
            cH2 = perturb_details(resize_exact(cH2, scale_rows, scale_cols))
            cV2 = perturb_details(resize_exact(cV2, scale_rows, scale_cols))
            cD2 = perturb_details(resize_exact(cD2, scale_rows, scale_cols))

            new_coeffs = [cA, (cH1, cV1, cD1), (cH2, cV2, cD2)]

        elif level == 3:
            cA, (cH1, cV1, cD1), (cH2, cV2, cD2), (cH3, cV3, cD3) = coeffs
            cA = resize_exact(cA, scale_rows, scale_cols)
            cH1 = perturb_details(resize_exact(cH1, scale_rows, scale_cols))
            cV1 = perturb_details(resize_exact(cV1, scale_rows, scale_cols))
            cD1 = perturb_details(resize_exact(cD1, scale_rows, scale_cols))
            cH2 = perturb_details(resize_exact(cH2, scale_rows, scale_cols))
            cV2 = perturb_details(resize_exact(cV2, scale_rows, scale_cols))
            cD2 = perturb_details(resize_exact(cD2, scale_rows, scale_cols))
            cH3 = perturb_details(resize_exact(cH3, scale_rows, scale_cols))
            cV3 = perturb_details(resize_exact(cV3, scale_rows, scale_cols))
            cD3 = perturb_details(resize_exact(cD3, scale_rows, scale_cols))

            new_coeffs = [cA, (cH1, cV1, cD1), (cH2, cV2, cD2), (cH3, cV3, cD3)]
        else:
            raise ValueError("Wavelet level must be 1, 2, or 3")

        try:
            reconstructed = pywt.waverec2(new_coeffs, wavelet_type)
        except ValueError:
            reconstructed = pywt.waverec2(new_coeffs, 'db1')

        new_start_row = int(block_row * block_size * scale_rows)
        new_start_col = int(block_col * block_size * scale_cols)

        threshold = 1e-10
        for r in range(min(reconstructed.shape[0], new_rows - new_start_row)):
            for c in range(min(reconstructed.shape[1], new_cols - new_start_col)):
                val = reconstructed[r, c]
                if abs(val) > threshold:
                    row_idx = new_start_row + r
                    col_idx = new_start_col + c
                    if row_idx < new_rows and col_idx < new_cols:
                        result_rows.append(row_idx)
                        result_cols.append(col_idx)
                        result_data.append(val)

    return sp.csr_matrix((result_data, (result_rows, result_cols)), shape=(new_rows, new_cols))