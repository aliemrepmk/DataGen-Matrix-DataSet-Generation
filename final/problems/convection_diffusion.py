import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from skfem import MeshTri, Basis, asm, solve, condense, LinearForm, BilinearForm
from skfem.element import ElementTriP1
from skfem.helpers import grad, dot
# from skfem.visuals.matplotlib import draw # Not used in this specific function's plotting
# Import the new utilities
from utils.plot_utils import plot_matrices_for_web, fig_to_base64_str
from utils.mesh_utils import generate_random_stone_mesh


# Renamed function
def solve_convection_diffusion_problem_for_web(mesh_type="rectangle", mesh_res=None,
                                               epsilon=None, b_field=None, f_value=None):
    """
    Solve -ε Δu + b·∇u = f with Dirichlet BCs using FEM.
    Returns solution and plots as base64 strings for web display.
    """
    generated_plots = [] # Initialize list for plot strings

    if mesh_type == 'random':
        print(f"[CONV-DIFF] Random mesh using generate_random_stone_mesh(). Res hint: {mesh_res}")
        mesh = generate_random_stone_mesh()
    elif mesh_type == 'circle':
        print(f"[CONV-DIFF] Using built-in circular mesh. Res hint: {mesh_res}")
        # Use skfem's built-in circle mesh
        nrefs = mesh_res if mesh_res is not None else 3
        mesh = MeshTri.init_circle(nrefs)
    elif mesh_type == 'rectangle':
        print(f"[CONV-DIFF] Using built-in rectangular mesh. Res hint: {mesh_res}")
        # Use skfem's built-in square mesh (default is unit square)
        nrefs = mesh_res if mesh_res is not None else 3
        # Create a square mesh and refine it
        mesh = MeshTri().refined(nrefs)
    elif mesh_type == 'lshaped':
        print(f"[CONV-DIFF] Using L-shaped domain mesh. Res hint: {mesh_res}")
        # Initialize L-shaped domain mesh
        mesh = MeshTri.init_lshaped()
        # Refine if needed
        if mesh_res is not None and mesh_res > 0:
            mesh = mesh.refined(mesh_res)
    elif mesh_type == 'symmetric':
        print(f"[CONV-DIFF] Using symmetric square mesh. Res hint: {mesh_res}")
        # Initialize the symmetric mesh of unit square
        mesh = MeshTri.init_symmetric()
        # Refine if needed
        if mesh_res is not None and mesh_res > 0:
            mesh = mesh.refined(mesh_res)
    elif mesh_type == 'sqsymmetric':
        print(f"[CONV-DIFF] Using square symmetric mesh. Res hint: {mesh_res}")
        # Initialize another variant of symmetric mesh
        mesh = MeshTri.init_sqsymmetric()
        # Refine if needed
        if mesh_res is not None and mesh_res > 0:
            mesh = mesh.refined(mesh_res)
    elif mesh_type == 'tensor':
        print(f"[CONV-DIFF] Using tensor product mesh. Res hint: {mesh_res}")
        # Determine the resolution based on the hint
        n_points = max(3, mesh_res if mesh_res is not None else 10)
        # Create a tensor product mesh with specified points in each dimension
        x = np.linspace(0, 1, n_points)
        y = np.linspace(0, 1, n_points)
        mesh = MeshTri.init_tensor(x, y)
    else:
        raise ValueError(f"Invalid mesh_type: {mesh_type}. Choose 'circle', 'rectangle', 'lshaped', 'symmetric', 'sqsymmetric', 'tensor', or 'random'.")


    basis = Basis(mesh, ElementTriP1(), intorder=3) # intorder might need adjustment based on element and terms

    # Parameter randomization or defaults
    epsilon_val = epsilon if epsilon is not None else np.random.uniform(1e-3, 5e-2) # Adjusted range for potentially more visible diffusion
    
    if b_field is None:
        # Randomize advection field components
        bx_val = np.random.uniform(0.5, 2.5) * (1 if np.random.rand() > 0.5 else -1) # Allow negative components
        by_val = np.random.uniform(0.5, 2.5) * (1 if np.random.rand() > 0.5 else -1)
        b_field_val = (bx_val, by_val)
    else:
        b_field_val = b_field
        
    f_value_val = f_value if f_value is not None else np.random.uniform(0.1, 2.0) # Adjusted range
    
    bx, by = b_field_val

    print(f"[CONV-DIFF] Effective Params: Mesh Vertices: {mesh.p.shape[1]}, ε: {epsilon_val:.3e} | b: ({bx:.2f}, {by:.2f}) | f: {f_value_val:.2f}")


    @BilinearForm
    def diffusion_form(u, v, w):
        return epsilon_val * dot(grad(u), grad(v))

    @BilinearForm
    def advection_form(u, v, w):
        # Ensure w.b_vec is passed if b varies spatially. Here b is constant.
        return (bx * grad(u)[0] + by * grad(u)[1]) * v

    # Assemble stiffness matrix
    A = asm(diffusion_form, basis) + asm(advection_form, basis)

    @LinearForm
    def rhs_form(v, w):
        return f_value_val * v

    # Assemble load vector
    F = asm(rhs_form, basis)

    # Identify boundary DOFs and apply zero Dirichlet conditions
    D = mesh.boundary_nodes() # Gets all nodes on the boundary
    # For ElementTriP1, boundary_nodes() directly gives DOFs on the boundary.
    
    # Condense system (apply boundary conditions)
    # expand=True by default in newer skfem if x0 not given for D.
    # To get u_full later, it's often easier to solve condensed and then expand.
    # Or use basis.zeros() and apply solve to A, F with D and x0.
    
    # u_full = basis.zeros() # Initialize solution vector with zeros (satisfies BC for D)
    # D_dofs = basis.get_dofs(D) # More explicit way to get DOFs if D is just node indices
    D_dofs = basis.get_dofs(mesh.boundary_nodes()).flatten()

    # Solve the system A u = F, applying BCs u[D] = 0
    # skfem.solve can handle this directly with D and x (Dirichlet values)
    try:
        # We provide A, F and then D for Dirichlet DOFs.
        # u_full will be modified in place by skfem.solve with these arguments.
        # For zero Dirichlet conditions, x (Dirichlet values) defaults to zero for DOFs in D.
        u_full = solve(*condense(A, F, D=D_dofs))
        # active_dofs = np.setdiff1d(np.arange(basis.N), D)
        # u_full[active_dofs] = solve(*condense(A, F, D=D))

    except Exception as e:
        print(f"Error during solve: {e}")
        # Handle error, maybe return empty plots or raise
        fig_err, ax_err = plt.subplots() # Create a new figure for the error message
        ax_err.text(0.5, 0.5, f"Solver Error:\n{e}", ha='center', va='center', wrap=True)
        ax_err.axis('off') # Hide axes for error text plot
        # Ensure generated_plots is defined before this point
        generated_plots.append(fig_to_base64_str(fig_err))
        return mesh, basis, basis.zeros(), generated_plots # Placeholder error plot


    # 1. Plot matrices A and F
    # Use the original (uncondensed) A and F for plotting matrix structure before BCs
    matrix_plot_strings = plot_matrices_for_web(A, F, title_suffix="(Convection–Diffusion)")
    generated_plots.extend(matrix_plot_strings)

    # 2. Plot the solution u_full
    fig_sol, ax_sol = plt.subplots(figsize=(8, 6))
    tri = mtri.Triangulation(mesh.p[0, :], mesh.p[1, :], mesh.t.T)
    contour = ax_sol.tricontourf(tri, u_full, cmap='hot', levels=20) # Added levels for smoother contour
    fig_sol.colorbar(contour, ax=ax_sol, label="Solution u")
    ax_sol.set_title("Convection–Diffusion: Solution Field")
    ax_sol.set_xlabel("X-coordinate")
    ax_sol.set_ylabel("Y-coordinate")
    ax_sol.set_aspect('equal', adjustable='box') # Ensure aspect ratio is equal
    
    plot_sol_b64 = fig_to_base64_str(fig_sol)
    generated_plots.append(plot_sol_b64)

    print(f"[CONV-DIFF-DEBUG] Number of plots generated: {len(generated_plots)}")
    if generated_plots:
        print(f"[CONV-DIFF-DEBUG] First plot string length: {len(generated_plots[0])}") # Should be large
    
    # Return original results along with the list of plot strings
    return mesh, basis, u_full, generated_plots