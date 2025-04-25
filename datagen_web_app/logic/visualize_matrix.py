import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def visualize_matrices(matrix1, matrix2, title1="Original Matrix", title2="Expanded Matrix", save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    scale_factor = matrix2.shape[0] / matrix1.shape[0]

    values1 = matrix1.data
    values2 = matrix2.data

    min_val1, max_val1 = np.min(values1), np.max(values1) if values1.size > 0 else (0, 0)
    min_val2, max_val2 = np.min(values2), np.max(values2) if values2.size > 0 else (0, 0)

    axes[0].spy(matrix1, markersize=1)
    axes[0].set_title(title1, fontsize=14)
    axes[0].set_xlabel(f"Nonzeros: {matrix1.nnz}\nMin: {min_val1:.2f}, Max: {max_val1:.2f}", fontsize=12)

    axes[1].spy(matrix2, markersize=1)
    axes[1].set_title(title2, fontsize=14)
    axes[1].set_xlabel(f"Nonzeros: {matrix2.nnz}\nMin: {min_val2:.2f}, Max: {max_val2:.2f}", fontsize=12)

    fig.suptitle(f"Expansion Factor: {scale_factor:.2f}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()

def visualize_heatmaps(
    matrix1,
    matrix2,
    title1="Original Matrix",
    title2="Expanded Matrix",
    save_path=None,
):
    """
    Create **one** PNG with two heat-maps (zeros -> white) side-by-side.
    Works for sparse or dense inputs.
    """

    # ── dense copies ──────────────────────────────────────────────
    m1 = matrix1.toarray() if hasattr(matrix1, "toarray") else np.asarray(matrix1)
    m2 = matrix2.toarray() if hasattr(matrix2, "toarray") else np.asarray(matrix2)

    # ── colour-bar limits per panel – like your spy version ───────
    def limits(mat):
        nz = mat[mat != 0]
        return (nz.min(), nz.max()) if nz.size else (0, 1)

    vmin1, vmax1 = limits(m1)
    vmin2, vmax2 = limits(m2)

    # zero → NaN so we can render them white with set_bad
    m1_masked = np.ma.masked_where(m1 == 0, m1)
    m2_masked = np.ma.masked_where(m2 == 0, m2)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im0 = axes[0].imshow(m1_masked, cmap=cmap,
                         norm=mcolors.Normalize(vmin=vmin1, vmax=vmax1),
                         interpolation="nearest")
    axes[0].set_title(title1, fontsize=14)
    axes[0].set_xlabel(f"Nonzeros: {matrix1.nnz}\nMin: {vmin1:.2f}, Max: {vmax1:.2f}",
                       fontsize=12)

    im1 = axes[1].imshow(m2_masked, cmap=cmap,
                         norm=mcolors.Normalize(vmin=vmin2, vmax=vmax2),
                         interpolation="nearest")
    axes[1].set_title(title2, fontsize=14)
    axes[1].set_xlabel(f"Nonzeros: {matrix2.nnz}\nMin: {vmin2:.2f}, Max: {vmax2:.2f}",
                       fontsize=12)

    scale_factor = matrix2.shape[0] / matrix1.shape[0]
    fig.suptitle(f"Expansion Factor: {scale_factor:.2f}",
                 fontsize=14, fontweight="bold")

    # make the two colour bars the same height
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()