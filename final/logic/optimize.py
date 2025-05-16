from typing import Dict, List, Optional, Union
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

# Weights for different matrix properties
weights = {
    # Row-based statistics
    "row_median_min": 1.5,
    "row_median_max": 1.5,
    "row_median_mean": 1.5,
    "row_median_std": 1.5,
    "row_mean_min": 0.8,
    "row_mean_max": 0.8,
    "row_mean_mean": 0.8,
    "row_mean_std": 0.8,
    "row_std_min": 0.7,
    "row_std_max": 0.7,
    "row_std_mean": 0.7,
    "row_std_std": 0.7,
    
    # Column-based statistics
    "col_median_min": 1.0,
    "col_median_max": 1.0,
    "col_median_mean": 1.0,
    "col_median_std": 1.0,
    "col_mean_min": 0.6,
    "col_mean_max": 0.6,
    "col_mean_mean": 0.6,
    "col_mean_std": 0.6,
    "col_std_min": 0.5,
    "col_std_max": 0.5,
    "col_std_mean": 0.5,
    "col_std_std": 0.5,
    
    # Structural properties
    "pattern_symmetry": 1.2,
    "numerical_symmetry": 1.2,
    "density": 1.2,
    "bandwidth": 0.01,
    "estimated_condition_number": 0.01,
    "norm_1": 0.01,
    "norm_inf": 0.01,
    "frobenius_norm": 0.01,
    "avg_distance_to_diagonal": 0.7,
    "num_diagonals_with_nonzeros": 0.2,
    "num_structurally_unsymmetric_elements": 0.5,
}

def compute_matrix_properties(matrix: sp.spmatrix) -> Dict[str, Union[float, int, bool]]:
    """Compute various properties of a sparse matrix."""
    props = {}
    
    # Basic dimensions and density
    num_rows, num_cols = matrix.shape
    props["num_rows"] = num_rows
    props["num_cols"] = num_cols
    nnz = matrix.nnz
    props["nnz"] = nnz
    total_elements = num_rows * num_cols
    props["density"] = float(nnz) / total_elements if total_elements > 0 else 0.0
    
    # Pattern symmetry
    rows, cols = matrix.nonzero()
    nonzero_positions = set(zip(rows, cols))
    t_rows, t_cols = matrix.T.nonzero()
    transpose_nonzero_positions = set(zip(t_rows, t_cols))
    props["pattern_symmetry"] = 1.0 if nonzero_positions == transpose_nonzero_positions else 0.0
    
    # Numerical symmetry
    diff = matrix - matrix.T
    diff_norm = np.sqrt((diff.data ** 2).sum())
    mat_norm = np.sqrt((matrix.data ** 2).sum())
    relative_symmetry = diff_norm / (mat_norm if mat_norm > 0 else 1)
    props["numerical_symmetry"] = 1.0 if relative_symmetry < 1e-6 else 0.0
    
    # Row-wise statistics
    row_means = np.empty(num_rows)
    row_stds = np.empty(num_rows)
    row_medians = np.empty(num_rows)
    
    for i in range(num_rows):
        start = matrix.indptr[i]
        end = matrix.indptr[i+1]
        data = matrix.data[start:end]
        nnz_row = end - start
        
        # Mean
        total = data.sum()
        row_mean = total / num_cols
        row_means[i] = row_mean
        
        # Standard deviation
        sq_sum = (data ** 2).sum() if nnz_row > 0 else 0.0
        E_X2 = sq_sum / num_cols
        variance = E_X2 - row_mean**2
        row_stds[i] = np.sqrt(variance) if variance > 0 else 0.0
        
        # Median
        num_zeros = num_cols - nnz_row
        if num_zeros > num_cols // 2:
            row_medians[i] = 0.0
        else:
            full_row = np.concatenate([data, np.zeros(num_zeros)])
            row_medians[i] = np.median(full_row)
    
    # Store row statistics
    props["row_mean_min"] = np.min(row_means)
    props["row_mean_max"] = np.max(row_means)
    props["row_mean_mean"] = np.mean(row_means)
    props["row_mean_std"] = np.std(row_means, ddof=1) if num_rows > 1 else 0.0
    
    props["row_std_min"] = np.min(row_stds)
    props["row_std_max"] = np.max(row_stds)
    props["row_std_mean"] = np.mean(row_stds)
    props["row_std_std"] = np.std(row_stds, ddof=1) if num_rows > 1 else 0.0
    
    props["row_median_min"] = np.min(row_medians)
    props["row_median_max"] = np.max(row_medians)
    props["row_median_mean"] = np.mean(row_medians)
    props["row_median_std"] = np.std(row_medians, ddof=1) if num_rows > 1 else 0.0
    
    # Column-wise statistics
    matrix_csc = matrix.tocsc()
    col_means = np.empty(num_cols)
    col_stds = np.empty(num_cols)
    col_medians = np.empty(num_cols)
    
    for j in range(num_cols):
        start = matrix_csc.indptr[j]
        end = matrix_csc.indptr[j+1]
        data = matrix_csc.data[start:end]
        nnz_col = end - start
        
        # Mean
        total = data.sum()
        col_mean = total / num_rows
        col_means[j] = col_mean
        
        # Standard deviation
        sq_sum = (data ** 2).sum() if nnz_col > 0 else 0.0
        E_X2 = sq_sum / num_rows
        variance = E_X2 - col_mean**2
        col_stds[j] = np.sqrt(variance) if variance > 0 else 0.0
        
        # Median
        num_zeros = num_rows - nnz_col
        if num_zeros > num_rows // 2:
            col_medians[j] = 0.0
        else:
            full_col = np.concatenate([data, np.zeros(num_zeros)])
            col_medians[j] = np.median(full_col)
    
    # Store column statistics
    props["col_mean_min"] = np.min(col_means)
    props["col_mean_max"] = np.max(col_means)
    props["col_mean_mean"] = np.mean(col_means)
    props["col_mean_std"] = np.std(col_means, ddof=1) if num_cols > 1 else 0.0
    
    props["col_std_min"] = np.min(col_stds)
    props["col_std_max"] = np.max(col_stds)
    props["col_std_mean"] = np.mean(col_stds)
    props["col_std_std"] = np.std(col_stds, ddof=1) if num_cols > 1 else 0.0
    
    props["col_median_min"] = np.min(col_medians)
    props["col_median_max"] = np.max(col_medians)
    props["col_median_mean"] = np.mean(col_medians)
    props["col_median_std"] = np.std(col_medians, ddof=1) if num_cols > 1 else 0.0
    
    # Bandwidth and diagonal properties
    if nnz > 0:
        distances = np.abs(np.array(rows) - np.array(cols))
        props["bandwidth"] = distances.max()
        props["avg_distance_to_diagonal"] = distances.mean()
        diag_offsets = np.array(rows) - np.array(cols)
        props["num_diagonals_with_nonzeros"] = len(np.unique(diag_offsets))
    else:
        props["bandwidth"] = 0
        props["avg_distance_to_diagonal"] = None
        props["num_diagonals_with_nonzeros"] = 0
    
    # Structural unsymmetry
    diff_positions = nonzero_positions.symmetric_difference(transpose_nonzero_positions)
    props["num_structurally_unsymmetric_elements"] = len(diff_positions)
    
    # Matrix norms
    abs_matrix = abs(matrix)
    col_sum = np.array(abs_matrix.sum(axis=0)).ravel()
    props["norm_1"] = col_sum.max() if col_sum.size > 0 else None
    row_sum = np.array(abs_matrix.sum(axis=1)).ravel()
    props["norm_inf"] = row_sum.max() if row_sum.size > 0 else None
    props["frobenius_norm"] = np.sqrt((matrix.data ** 2).sum())
    
    # Condition number estimation
    if min(num_rows, num_cols) > 1:
        try:
            u, s, vt = svds(matrix, k=2, which='LM')
            s = np.sort(s)
            smallest, largest = s[0], s[-1]
            cond_est = largest / smallest if smallest > 0 else np.inf
        except Exception:
            cond_est = None
    else:
        cond_est = 1.0 if nnz > 0 else None
    props["estimated_condition_number"] = cond_est
    
    return props

def compute_scaling_params(list_of_prop_dicts: List[Dict[str, Union[float, int, bool]]], prop_names: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    """Compute scaling parameters for matrix properties."""
    scaling_dict = {}
    for prop in prop_names:
        all_vals = []
        for props in list_of_prop_dicts:
            val = props.get(prop, None)
            if val is not None and isinstance(val, (int, float)):
                all_vals.append(val)
        if len(all_vals) == 0:
            scaling_dict[prop] = {"min": None, "max": None}
        else:
            scaling_dict[prop] = {"min": min(all_vals), "max": max(all_vals)}
    return scaling_dict

def compute_property_loss_minmax(
    original_props: Dict[str, Union[float, int, bool]],
    new_props: Dict[str, Union[float, int, bool]],
    weights: Dict[str, float],
    scaling_dict: Dict[str, Dict[str, Optional[float]]]
) -> float:
    """Compute property-based loss using min-max scaling."""
    loss = 0.0
    epsilon = 1e-12
    
    for prop_name, w in weights.items():
        orig_val = original_props.get(prop_name, None)
        new_val = new_props.get(prop_name, None)
        
        if orig_val is None or new_val is None:
            continue
        
        smin = scaling_dict[prop_name]["min"]
        smax = scaling_dict[prop_name]["max"]
        
        if smin is not None and smax is not None and smax != smin:
            scaled_orig_val = (orig_val - smin) / (smax - smin + epsilon)
            scaled_new_val = (new_val - smin) / (smax - smin + epsilon)
        else:
            scaled_orig_val = 0.0
            scaled_new_val = 0.0
        
        diff = np.abs(scaled_orig_val - scaled_new_val)
        loss += w * diff
    
    return loss / 100

def cost_function(original_matrix: sp.spmatrix, candidate_matrix: sp.spmatrix, weights: Dict[str, float], scaling_dict: Dict[str, Dict[str, Optional[float]]]) -> float:
    """Compute the cost between original and candidate matrices."""
    original_props = compute_matrix_properties(original_matrix)
    candidate_props = compute_matrix_properties(candidate_matrix)
    return compute_property_loss_minmax(original_props, candidate_props, weights, scaling_dict)

def cost_function_raw(original_matrix, candidate_matrix, weights):
    """Compute the cost between original and candidate matrices using raw (unscaled) property differences."""
    original_props = compute_matrix_properties(original_matrix)
    candidate_props = compute_matrix_properties(candidate_matrix)
    loss = 0.0
    for prop, w in weights.items():
        orig_val = original_props.get(prop, 0)
        cand_val = candidate_props.get(prop, 0)
        if orig_val is not None and cand_val is not None:
            loss += w * abs(orig_val - cand_val)
    return loss / 100

def cost_function_normalized(original_matrix, candidate_matrix, weights):
    original_props = compute_matrix_properties(original_matrix)
    candidate_props = compute_matrix_properties(candidate_matrix)
    loss = 0.0
    for prop, w in weights.items():
        orig_val = original_props.get(prop, 0)
        cand_val = candidate_props.get(prop, 0)
        if orig_val is not None and cand_val is not None:
            norm = abs(orig_val) if abs(orig_val) > 1e-8 else 1.0
            loss += w * abs(orig_val - cand_val) / norm
    return loss

def perturb_values(matrix: sp.spmatrix, epsilon: float = 0.01) -> sp.spmatrix:
    """Perturb matrix values by a small random factor."""
    perturbed = matrix.copy()
    perturbation = np.random.uniform(1 - epsilon, 1 + epsilon, size=perturbed.data.shape)
    perturbed.data *= perturbation
    return perturbed

def perturb_positions(matrix: sp.spmatrix, max_offset: int = 1, perturb_fraction: float = 0.1) -> sp.spmatrix:
    """Perturb matrix element positions by a small random offset."""
    coo = matrix.tocoo()
    num_entries = coo.data.size
    num_to_perturb = int(perturb_fraction * num_entries)
    indices_to_perturb = np.random.choice(num_entries, num_to_perturb, replace=False)
    
    new_rows = coo.row.copy()
    new_cols = coo.col.copy()
    
    row_offsets = np.random.randint(-max_offset, max_offset + 1, size=num_to_perturb)
    col_offsets = np.random.randint(-max_offset, max_offset + 1, size=num_to_perturb)
    
    new_rows[indices_to_perturb] = np.clip(new_rows[indices_to_perturb] + row_offsets, 0, matrix.shape[0] - 1)
    new_cols[indices_to_perturb] = np.clip(new_cols[indices_to_perturb] + col_offsets, 0, matrix.shape[1] - 1)
    
    return sp.coo_matrix((coo.data, (new_rows, new_cols)), shape=matrix.shape).tocsr()

def optimize_matrix(matrix: sp.spmatrix, epsilon: float = 0.09, max_offset: int = 1, perturb_fraction: float = 0.1) -> sp.spmatrix:
    """Optimize matrix by perturbing values and positions."""
    matrix = perturb_values(matrix, epsilon)
    matrix = perturb_positions(matrix, max_offset, perturb_fraction)
    return matrix