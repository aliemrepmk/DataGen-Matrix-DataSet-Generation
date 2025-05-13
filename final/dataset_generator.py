from logic.load_matrix import load_matrix
from logic.bilinear    import scale_sparse_matrix_bilinear
from logic.dct         import scale_sparse_matrix_dct_blockwise
from logic.fourier     import scale_sparse_matrix_fourier
from logic.gaussian    import scale_sparse_matrix_gaussian
from logic.graph       import scale_sparse_matrix_graph
from logic.image       import scale_sparse_matrix_image
from logic.lanczos     import scale_sparse_matrix_lanczos
from logic.nearest     import scale_sparse_matrix_nearest
from logic.wavelet     import scale_sparse_matrix_wavelet
import random

downscale = [
    scale_sparse_matrix_bilinear,
    scale_sparse_matrix_dct_blockwise,
    scale_sparse_matrix_fourier,
    scale_sparse_matrix_gaussian,
    scale_sparse_matrix_graph,
    scale_sparse_matrix_image,
    scale_sparse_matrix_lanczos,
    scale_sparse_matrix_nearest,
    scale_sparse_matrix_wavelet
]

upscale = [
    scale_sparse_matrix_bilinear,
    scale_sparse_matrix_dct_blockwise,
    scale_sparse_matrix_fourier,
    scale_sparse_matrix_graph,
    scale_sparse_matrix_image,
    scale_sparse_matrix_nearest,
    scale_sparse_matrix_wavelet
]

def get_matrix_generation_params():
    """
    Ask the user for the number of matrices, minimum dimension, and maximum dimension, performing basic validation.

    Returns:
        tuple: A tuple containing (num_matrices, min_dim, max_dim)
    """
    num_matrices = None
    min_dim = None
    max_dim = None

    # Get the number of matrices
    while num_matrices is None:
        try:
            input_str = input("How many matrices do you want to generate? ")
            num_matrices = int(input_str)
            if num_matrices <= 0:
                print("Error: Number of matrices must be a positive integer (greater than 0).")
                num_matrices = None # Reset to loop again
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    # Get the minimum dimension
    while min_dim is None:
        try:
            input_str = input("Enter the minimum dimension (e.g., rows/columns): ")
            min_dim = int(input_str)
            if min_dim <= 0:
                print("Error: Minimum dimension must be a positive integer (greater than 0).")
                min_dim = None # Reset to loop again
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    # Get the maximum dimension
    while max_dim is None:
        try:
            input_str = input(f"Enter the maximum dimension (must be >= {min_dim}): ")
            max_dim = int(input_str)
            if max_dim <= 0:
                 print("Error: Maximum dimension must be a positive integer (greater than 0).")
                 max_dim = None # Reset to loop again
            elif max_dim < min_dim:
                print(f"Error: Maximum dimension ({max_dim}) cannot be less than minimum dimension ({min_dim}).")
                max_dim = None # Reset to loop again
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    return num_matrices, min_dim, max_dim

#scale_sparse_matrix_bilinear       (original_matrix: sp.csr_matrix, new_size: int, match_nnz = True) -> sp.csr_matrix:
#scale_sparse_matrix_dct_blockwise  (original_matrix: sp.csr_matrix, new_size: int, block_size: int = 8, thresh: float = 1e-8) -> sp.csr_matrix:
#scale_sparse_matrix_fourier        (original_matrix: sp.csr_matrix, new_size: int) -> sp.csr_matrix:
#scale_sparse_matrix_gaussian       (original_matrix: sp.csr_matrix, new_size: int, sigma: float = 1.0) -> sp.csr_matrix:
#scale_sparse_matrix_graph          (original_matrix: sp.csr_matrix, new_size: int, match_nnz: bool = True):
#scale_sparse_matrix_image          (original_matrix: sp.csr_matrix, new_size: int, resize_method: int = Image.BOX) -> sp.csr_matrix:
#scale_sparse_matrix_lanczos        (original_matrix: sp.csr_matrix, new_size: int, match_nnz: bool = True, a: int = 3) -> sp.csr_matrix:
#scale_sparse_matrix_nearest        (original_matrix: sp.csr_matrix, new_size: int, match_nnz: bool = True) -> sp.csr_matrix:
#scale_sparse_matrix_wavelet        (original_matrix: sp.csr_matrix, new_size: int, wavelet_type: str = 'db1', level: int = 2) -> sp.csr_matrix:

if __name__ == "__main__":
    input_matrix = load_matrix("matrix/cage7.mtx")
    print(input_matrix.shape[0], input_matrix.shape[1], input_matrix.nnz)

    print("Please provide details for matrix generation:")
    try:
        number_of_operations, minimum_dimension, maximum_dimension = get_matrix_generation_params()

        print("\n--- Configuration Summary ---")
        print(f"Number of matrices to generate: {number_of_operations}")
        print(f"Minimum matrix dimension:       {minimum_dimension}")
        print(f"Maximum matrix dimension:       {maximum_dimension}")

        upscale_count = 0
        downscale_count = 0
        downscaled_matrices = []
        upscaled_matrices = []
        all_generated_matrices = []

        for i in range(number_of_operations):
            print(f"\n--- Operation {i+1}/{number_of_operations} ---")
            operation = random.randint(0, 1)

            if operation == 0: #downscale
                selected_func = random.choice(downscale)
                new_dim = random.randint(minimum_dimension, input_matrix.shape[0])
                while(selected_func == scale_sparse_matrix_wavelet or selected_func == scale_sparse_matrix_bilinear):
                    selected_func = random.choice(downscale)
                    """
                    if(new_dim % 2 != 0):
                        new_dim += 1
                    """
                print("Downscale --- Selected method: " + str(selected_func) + ", new dimensions: " + str(new_dim) + "x" + str(new_dim))

                downcaled_matrix = selected_func(input_matrix, new_dim)
                downscaled_matrices.append(downcaled_matrix)
                downscale_count += 1
                
            else: #upscale
                selected_func = random.choice(upscale)
                new_dim = random.randint(input_matrix.shape[0], maximum_dimension)
                while(selected_func == scale_sparse_matrix_wavelet or selected_func == scale_sparse_matrix_bilinear):
                    selected_func = random.choice(upscale)
                    """
                    if(new_dim % 2 != 0):
                        new_dim += 1
                    """
                print("Upscale --- Selected method: " + str(selected_func) + ", new dimensions: " + str(new_dim) + "x" + str(new_dim))

                upscaled_matrix = selected_func(input_matrix, new_dim)
                upscaled_matrices.append(upscaled_matrix)
                upscale_count += 1

        print("Upscale: " + str(upscale_count) + " , Downscale: " + str(downscale_count))
        print("Number of matrices generated: " + str(len(downscaled_matrices) + len(upscaled_matrices)))

    except KeyboardInterrupt:
        print("\nOperation cancelled.")