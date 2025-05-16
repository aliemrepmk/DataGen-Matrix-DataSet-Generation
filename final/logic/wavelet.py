import numpy as np
import scipy.sparse as sp
import pywt
from skimage.transform import resize

def get_scaled_shape(matrix, scale_rows, scale_cols):
    h, w = matrix.shape
    return (int(round(h * scale_rows)), int(round(w * scale_cols)))

def resize_exact(matrix, scale_rows, scale_cols):
    target_shape = get_scaled_shape(matrix, scale_rows, scale_cols)
    return resize(matrix, output_shape=target_shape, order=0, mode='constant', cval=0, anti_aliasing=False, preserve_range=True)

def scale_sparse_matrix_wavelet(original_matrix: sp.csr_matrix, new_size: int, wavelet_type: str = 'db1', level: int = 2) -> sp.csr_matrix:
    # Ensure new_size is even for wavelet compatibility
    new_size = 2 * (new_size // 2)
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
    scale_factor = new_size / max(orig_rows, orig_cols)  # Use max to maintain aspect ratio

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
            cA = resize_exact(cA, scale_factor, scale_factor)
            cH = resize_exact(cH, scale_factor, scale_factor)
            cV = resize_exact(cV, scale_factor, scale_factor)
            cD = resize_exact(cD, scale_factor, scale_factor)
            new_coeffs = [cA, (cH, cV, cD)]

        elif level == 2:
            cA, (cH1, cV1, cD1), (cH2, cV2, cD2) = coeffs
            cA = resize_exact(cA, scale_factor, scale_factor)
            cH1 = resize_exact(cH1, scale_factor, scale_factor)
            cV1 = resize_exact(cV1, scale_factor, scale_factor)
            cD1 = resize_exact(cD1, scale_factor, scale_factor)
            cH2 = resize_exact(cH2, scale_factor, scale_factor)
            cV2 = resize_exact(cV2, scale_factor, scale_factor)
            cD2 = resize_exact(cD2, scale_factor, scale_factor)

            new_coeffs = [cA, (cH1, cV1, cD1), (cH2, cV2, cD2)]

        elif level == 3:
            cA, (cH1, cV1, cD1), (cH2, cV2, cD2), (cH3, cV3, cD3) = coeffs
            cA = resize_exact(cA, scale_factor, scale_factor)
            cH1 = resize_exact(cH1, scale_factor, scale_factor)
            cV1 = resize_exact(cV1, scale_factor, scale_factor)
            cD1 = resize_exact(cD1, scale_factor, scale_factor)
            cH2 = resize_exact(cH2, scale_factor, scale_factor)
            cV2 = resize_exact(cV2, scale_factor, scale_factor)
            cD2 = resize_exact(cD2, scale_factor, scale_factor)
            cH3 = resize_exact(cH3, scale_factor, scale_factor)
            cV3 = resize_exact(cV3, scale_factor, scale_factor)
            cD3 = resize_exact(cD3, scale_factor, scale_factor)

            new_coeffs = [cA, (cH1, cV1, cD1), (cH2, cV2, cD2), (cH3, cV3, cD3)]
        else:
            raise ValueError("Wavelet level must be 1, 2, or 3")

        try:
            reconstructed = pywt.waverec2(new_coeffs, wavelet_type)
        except ValueError:
            reconstructed = pywt.waverec2(new_coeffs, 'db1')

        new_start_row = int(block_row * block_size * scale_factor)
        new_start_col = int(block_col * block_size * scale_factor)

        threshold = 1e-10
        for r in range(min(reconstructed.shape[0], new_size - new_start_row)):
            for c in range(min(reconstructed.shape[1], new_size - new_start_col)):
                val = reconstructed[r, c]
                if abs(val) > threshold:
                    row_idx = new_start_row + r
                    col_idx = new_start_col + c
                    if row_idx < new_size and col_idx < new_size:
                        result_rows.append(row_idx)
                        result_cols.append(col_idx)
                        result_data.append(val)

    return sp.csr_matrix((result_data, (result_rows, result_cols)), shape=(new_size, new_size))