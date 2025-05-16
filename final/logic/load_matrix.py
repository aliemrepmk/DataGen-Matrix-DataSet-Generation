import scipy.io as sio
import scipy.sparse as sp

def load_matrix(file_path):
    """Load a matrix from a .mtx file and ensure it's in CSR format with valid dtype."""
    print(f"Loading matrix from {file_path}...")
    try:
        matrix = sio.mmread(file_path)

        if not sp.isspmatrix(matrix):
            matrix = sp.csr_matrix(matrix)
        else:
            matrix = matrix.tocsr()

        # Check if dtype is valid for sparse matrices
        if matrix.dtype.kind == 'O':  # 'O' stands for object
            raise ValueError(f"Matrix has unsupported dtype=object (from file: {file_path})")

        print(f"{file_path} loaded successfully")
        return matrix
    except Exception as e:
        print(f"Error loading the matrix from {file_path}: {e}")
        return None