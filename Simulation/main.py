# Directory: my_fem_project/

# ---------------- main.py ----------------
from problems.fluid import solve_fluid_symmetric, solve_fluid_symmetric_custom, solve_fluid_nonsymmetric
from problems.heat import solve_heat
from problems.solid import solve_solid
from problems.convection_diffusion import solve_convection_diffusion_problem
from problems.minimal_surface import solve_minimal_surface
from problems.block_preconditioned_stokes import solve_block_preconditioned_stokes
import numpy as np
import random
import sys

if __name__ == "__main__":
    choice = input("Choose problem type ('fluid', 'heat', 'solid','minimal-surface','convection-diffusion','gaus and wavelet','stokes-3d'): ").strip().lower()
    if choice.startswith('f'):
        print("\nYou have selected the Fluid Mechanics problem.")
        print("Do you want a symmetric or non-symmetric approach?")
        print(" (Enter 's' for symmetric, 'n' for non-symmetric)")
        approach = input().strip().lower()

        if approach.startswith('s'):
            try:
                num_matrices = int(input("Enter how many different matrices to generate: ").strip())
            except Exception:
                print("Invalid input. Exiting.")
                sys.exit(1)

            mesh_choice_input = input("Enter mesh type ('circle', 'rectangle', or 'random'): ").strip().lower()
            print(f"\nGenerating {num_matrices} different symmetric fluid matrices...")

            for i in range(num_matrices):
                if mesh_choice_input == "random":
                    mesh_type = random.choice(["circle", "rectangle"])
                else:
                    mesh_type = mesh_choice_input

                mesh_resolution = np.random.randint(3, 7) if mesh_type == "circle" else np.random.randint(20, 51)
                viscosity = np.random.uniform(5.0, 20.0)
                reg_param = np.random.uniform(1e-6, 5e-6)

                print(f"\n--- Generating matrix set {i + 1} ---")
                print(f"Parameters: mesh_type={mesh_type}, mesh_resolution={mesh_resolution}, viscosity={viscosity:.2f}, reg_param={reg_param:.2e}")

                K, f, *_ = solve_fluid_symmetric_custom(mesh_type, mesh_resolution, viscosity, reg_param)

        elif approach.startswith('n'):
            print("\nRunning the non-symmetric fluid mechanics solver...\n")
            solve_fluid_nonsymmetric()
        else:
            solve_fluid_symmetric()

    elif choice.startswith('h'):
        solve_heat()

    elif choice.startswith('s'):
        solve_solid()

    elif choice.startswith('m'):
        solve_minimal_surface()

    elif choice.startswith('c'):
        solve_convection_diffusion_problem()

    elif choice.startswith('g'):
        import numpy as np
        data = np.random.rand(516, 516)
        solve_wavelet_transform(data, wavelet_name="db1", levels=2)
        solve_gaussian_pyramid(data, num_levels=3)

    elif choice.startswith('3'):
        solve_block_preconditioned_stokes(mesh_type='random')

    else:
        print("Unknown choice. Please enter 'fluid', 'heat', 'solid', 'minimal-surface', 'convection-diffusion', or 'gaus and wavelet'.")