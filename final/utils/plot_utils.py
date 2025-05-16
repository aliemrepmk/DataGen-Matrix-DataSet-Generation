# utils/plot_utils.py
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
import io
import base64

def fig_to_base64_str(fig):
    """Converts a matplotlib Figure to a base64 encoded string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)  # Close the figure to free memory
    return img_str

def plot_matrices_for_web(A, b, title_suffix=""):
    """
    Generates sparsity plot for matrix A and a plot for vector b,
    returning them as a list of base64 encoded image strings.
    """
    plot_strings = []
    
    # Ensure A is in a sparse format that spy can use efficiently if it's not already
    A_sparse = csc_matrix(A) 

    # Plot 1: Sparsity of A
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.spy(A_sparse, markersize=1)
    ax1.set_title("Sparsity Pattern of A " + title_suffix)
    ax1.set_xlabel("Column Index")
    ax1.set_ylabel("Row Index")
    plot_strings.append(fig_to_base64_str(fig1))

    # Plot 2: RHS vector b
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(b, "o")
    ax2.set_title("RHS vector b " + title_suffix)
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Value")
    ax2.grid(True)
    plot_strings.append(fig_to_base64_str(fig2))

    return plot_strings
