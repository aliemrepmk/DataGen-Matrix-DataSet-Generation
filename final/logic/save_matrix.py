import os
import scipy.sparse as sp
import scipy.io as sio

def save_matrix(matrix: sp.csr_matrix, file_name: str, folder_path: str) -> None:
    """
    Saves a SciPy CSR (Compressed Sparse Row) matrix to a file in Matrix Market (.mtx) format.
    """
    # Validate input types
    if not isinstance(matrix, sp.csr_matrix):
        raise TypeError("Input 'matrix' must be a sp.csr_matrix.")
    if not isinstance(file_name, str):
        raise TypeError("Input 'file_name' must be a string.")
    if not isinstance(folder_path, str):
        raise TypeError("Input 'folder_path' must be a string.")

    # Ensure the output folder exists
    if not os.path.exists(folder_path):
        try:
            # Attempt to create the directory if it doesn't exist
            os.makedirs(folder_path)
            print(f"Created directory: {folder_path}")
        except OSError as e:
            raise FileNotFoundError(
                f"The folder_path '{folder_path}' does not exist and could not be created. Error: {e}"
            ) from e

    # Construct the full file path
    if not file_name.endswith(".mtx"):
        output_filename = f"{file_name}.mtx"
    else:
        output_filename = file_name

    full_file_path = os.path.join(folder_path, output_filename)

    try:
        sio.mmwrite(target=full_file_path, a=matrix)
        print(f"Matrix successfully saved to: {full_file_path}")
    except Exception as e:
        # Catch any other errors during the mmwrite process
        print(f"An error occurred while saving the matrix: {e}")
        raise # Re-raise the exception after printing the message