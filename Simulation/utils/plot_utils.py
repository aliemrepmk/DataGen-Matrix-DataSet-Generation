import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix

def plot_matrices(A, b, title_suffix=""):
    A_sparse = csc_matrix(A)
    plt.figure(figsize=(8, 6))
    plt.spy(A_sparse, markersize=1)
    plt.title("Sparsity Pattern of A " + title_suffix)
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.show()


    plt.figure(figsize=(8, 3))
    plt.plot(b, "o")
    plt.title("RHS vector b " + title_suffix)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()
