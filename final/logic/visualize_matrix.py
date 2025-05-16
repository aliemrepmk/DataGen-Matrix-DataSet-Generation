import os
import matplotlib
matplotlib.use('Agg')  # Prevents GUI backend issues in some environments

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import networkx as nx  # Use the main networkx module
from scipy.sparse import csr_matrix

TEMP_VISUALIZATION_FOLDER = os.path.join(os.getcwd(), 'static', 'visualizations')
os.makedirs(TEMP_VISUALIZATION_FOLDER, exist_ok=True)

def visualize_matrices(matrix1, matrix2, title1="Original Matrix", title2="Expanded Matrix"):
    """
    Visualize the sparsity pattern of two matrices side-by-side using matplotlib's spy plot.
    Save the visualization as an image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].spy(matrix1, markersize=1)
    axes[0].set_title(title1, fontsize=14)

    axes[1].spy(matrix2, markersize=1)
    axes[1].set_title(title2, fontsize=14)

    plt.tight_layout()

    # Save the plot as an image
    filepath = os.path.join(TEMP_VISUALIZATION_FOLDER, 'spy_plot.png')
    plt.savefig(filepath)
    plt.close(fig)
    return filepath

def visualize_heatmaps(matrix1, matrix2, title1="Original Matrix", title2="Expanded Matrix"):
    """
    Visualize two matrices side-by-side as heatmaps. Save the visualization as an image.
    """
    m1 = matrix1.toarray() if hasattr(matrix1, "toarray") else np.asarray(matrix1)
    m2 = matrix2.toarray() if hasattr(matrix2, "toarray") else np.asarray(matrix2)

    def limits(mat):
        nz = mat[mat != 0]
        return (nz.min(), nz.max()) if nz.size else (0, 1)

    vmin1, vmax1 = limits(m1)
    vmin2, vmax2 = limits(m2)

    m1_masked = np.ma.masked_where(m1 == 0, m1)
    m2_masked = np.ma.masked_where(m2 == 0, m2)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im0 = axes[0].imshow(m1_masked, cmap=cmap, norm=mcolors.Normalize(vmin=vmin1, vmax=vmax1), interpolation="nearest")
    axes[0].set_title(title1, fontsize=14)

    im1 = axes[1].imshow(m2_masked, cmap=cmap, norm=mcolors.Normalize(vmin=vmin2, vmax=vmax2), interpolation="nearest")
    axes[1].set_title(title2, fontsize=14)

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Save the plot as an image
    filepath = os.path.join(TEMP_VISUALIZATION_FOLDER, 'heatmap.png')
    plt.savefig(filepath)
    plt.close(fig)
    return filepath

def visualize_graphs(matrix1: csr_matrix, matrix2: csr_matrix, title1="Original Matrix", title2="Expanded Matrix", node_limit=2000):
    """
    Visualize two CSR matrices as graphs side-by-side using networkx and matplotlib.
    Save the visualization as an image.
    """
    if matrix1.shape[0] > node_limit or matrix2.shape[0] > node_limit:
        raise ValueError(f"One of the matrices exceeds the node limit of {node_limit}. Visualization skipped.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, matrix, title in zip(axes, [matrix1, matrix2], [title1, title2]):
        G = nx.from_scipy_sparse_matrix(matrix)  # Use the correct function
        pos = nx.spring_layout(G, iterations=100, seed=42)

        weights = np.array([d['weight'] for _, _, d in G.edges(data=True)])
        colors = weights / weights.max() if weights.max() != 0 else weights

        nx.draw_networkx_nodes(G, pos, node_size=10, node_color='white', edgecolors='black', ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color=colors, edge_cmap=plt.cm.plasma, width=1.0, ax=ax)

        ax.set_title(f"{title}\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges", fontsize=12)
        ax.axis('off')

    plt.tight_layout()

    # Save the plot as an image
    filepath = os.path.join(TEMP_VISUALIZATION_FOLDER, 'graph.png')
    plt.savefig(filepath)
    plt.close(fig)
    return filepath