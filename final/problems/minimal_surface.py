import numpy as np
import matplotlib.pyplot as plt # Keep for direct figure creation, especially 3D
from skfem import MeshTri, Basis, asm, solve, condense, LinearForm, BilinearForm
from skfem.helpers import grad, dot
from skfem.element import ElementTriP1
# Import plot3 with an alias, remove show as it's blocking
from skfem.visuals.matplotlib import plot3 as skfem_plot3
# Import the new utilities
from utils.plot_utils import plot_matrices_for_web, fig_to_base64_str
from utils.mesh_utils import generate_random_stone_mesh


def solve_minimal_surface_for_web(mesh_type='rectangle', mesh_resolution_hint=None, max_iters=30, tolerance=1e-7, relaxation=0.7):
    """
    Solve the nonlinear minimal surface equation using Newton's method.
    Returns the solution and plots as base64 strings for web display.
    """
    generated_plots = []

    if mesh_type == 'random':
        # generate_random_stone_mesh will pick its own details
        print(f"[MIN-SURFACE] Random mesh type by generate_random_stone_mesh(). Res hint: {mesh_resolution_hint}")
        mesh = generate_random_stone_mesh()
    elif mesh_type == 'circle':
        print(f"[MIN-SURFACE] Using built-in circular mesh. Res hint: {mesh_resolution_hint}")
        # Use skfem's built-in circle mesh
        nrefs = mesh_resolution_hint if mesh_resolution_hint is not None else 3
        mesh = MeshTri.init_circle(nrefs)
    elif mesh_type == 'rectangle':
        print(f"[MIN-SURFACE] Using built-in rectangular mesh. Res hint: {mesh_resolution_hint}")
        # Use skfem's built-in square mesh (default is unit square)
        nrefs = mesh_resolution_hint if mesh_resolution_hint is not None else 3
        # Create a square mesh and refine it
        mesh = MeshTri().refined(nrefs)
    elif mesh_type == 'lshaped':
        print(f"[MIN-SURFACE] Using L-shaped domain mesh. Res hint: {mesh_resolution_hint}")
        # Initialize L-shaped domain mesh
        mesh = MeshTri.init_lshaped()
        # Refine if needed
        if mesh_resolution_hint is not None and mesh_resolution_hint > 0:
            mesh = mesh.refined(mesh_resolution_hint)
    elif mesh_type == 'symmetric':
        print(f"[MIN-SURFACE] Using symmetric square mesh. Res hint: {mesh_resolution_hint}")
        # Initialize the symmetric mesh of unit square
        mesh = MeshTri.init_symmetric()
        # Refine if needed
        if mesh_resolution_hint is not None and mesh_resolution_hint > 0:
            mesh = mesh.refined(mesh_resolution_hint)
    elif mesh_type == 'sqsymmetric':
        print(f"[MIN-SURFACE] Using square symmetric mesh. Res hint: {mesh_resolution_hint}")
        # Initialize another variant of symmetric mesh
        mesh = MeshTri.init_sqsymmetric()
        # Refine if needed
        if mesh_resolution_hint is not None and mesh_resolution_hint > 0:
            mesh = mesh.refined(mesh_resolution_hint)
    elif mesh_type == 'tensor':
        print(f"[MIN-SURFACE] Using tensor product mesh. Res hint: {mesh_resolution_hint}")
        # Determine the resolution based on the hint
        n_points = max(3, mesh_resolution_hint if mesh_resolution_hint is not None else 10)
        # Create a tensor product mesh with specified points in each dimension
        x = np.linspace(0, 1, n_points)
        y = np.linspace(0, 1, n_points)
        mesh = MeshTri.init_tensor(x, y)
    else:
        raise ValueError(f"Invalid mesh_type: {mesh_type}. Use 'circle', 'rectangle', 'lshaped', 'symmetric', 'sqsymmetric', 'tensor', or 'random'.")

    basis = Basis(mesh, ElementTriP1(), intorder=2) # intorder for P1 elements

    # Initial guess for solution x (height z)
    x_solution = basis.zeros() # u is traditionally solution, but here problem uses x
    
    # Define boundary conditions (Dirichlet)
    # x = sin(pi * p_x) on the boundary
    boundary_dofs = mesh.boundary_nodes() # For P1 elements, nodes are DOFs
    x_solution[boundary_dofs] = np.sin(np.pi * mesh.p[0, boundary_dofs])

    print(f"[MIN-SURFACE] Effective Params: Mesh Vertices: {mesh.p.shape[1]}, Max Iters: {max_iters}, Tol: {tolerance:.1e}, Relax: {relaxation}")

    # Define the nonlinear forms for Newton's method
    # Jacobian (bilinear form for K_T du = -R)
    @BilinearForm
    def jacobian_form(u, v, w): # u is du (increment), v is test_func, w has previous solution w_val=x_prev
        x_prev_interpolated = w['prev_sol_interpolated'] # Previous solution u_k (here denoted x_k)
        grad_x_prev = grad(x_prev_interpolated)
        norm_grad_x_prev_sq = 1 + dot(grad_x_prev, grad_x_prev) # (1 + |∇x_k|^2)
        
        term1 = dot(grad(u), grad(v)) / np.sqrt(norm_grad_x_prev_sq)
        term2 = (dot(grad(u), grad_x_prev) * dot(grad_x_prev, grad(v))) / (norm_grad_x_prev_sq**1.5)
        return term1 - term2

    # Residual (linear form for -R)
    @LinearForm
    def residual_form(v, w): # v is test_func, w has previous solution w_val=x_prev
        x_prev_interpolated = w['prev_sol_interpolated']
        grad_x_prev = grad(x_prev_interpolated)
        norm_grad_x_prev_sq = 1 + dot(grad_x_prev, grad_x_prev)
        # Original RHS form was F(u_k)(v) = integral ( (∇u_k . ∇v) / sqrt(1+|∇u_k|^2) )
        # We need -F(u_k)(v) for the Newton step J du = -F
        return - dot(grad_x_prev, grad(v)) / np.sqrt(norm_grad_x_prev_sq)


    # Newton's iteration
    J_matrix = None # To store the last Jacobian for plotting
    F_vector = None # To store the last residual vector for plotting

    for itr in range(max_iters):
        # Interpolate current solution x_solution to quadrature points for use in forms
        x_prev_interpolated_at_quad = basis.interpolate(x_solution)
        
        # Assemble Jacobian J and residual vector -F (RHS of Newton step)
        J_matrix = asm(jacobian_form, basis, prev_sol_interpolated=x_prev_interpolated_at_quad)
        # The 'rhs' form in original code was actually F, so for J du = -F, we use -F from asm(rhs_form)
        # The provided residual_form already includes the minus sign for -F.
        neg_F_vector = asm(residual_form, basis, prev_sol_interpolated=x_prev_interpolated_at_quad)

        # Store last F for plotting (neg_F_vector is -F, so F = -neg_F_vector)
        F_vector = -neg_F_vector 
        
        x_solution_old_iter = x_solution.copy()
        
        # Solve J_matrix * dx = neg_F_vector for dx, applying homogeneous Dirichlet for increment dx
        # Boundary DOFs D are for x_solution itself. The increment dx should be 0 on these DOFs.
        dx = solve(*condense(J_matrix, neg_F_vector, D=boundary_dofs))
        
        x_solution += relaxation * dx # Update solution: x_k+1 = x_k + relax * dx
        
        # Enforce boundary conditions on the updated solution (important!)
        x_solution[boundary_dofs] = np.sin(np.pi * mesh.p[0, boundary_dofs])
        
        norm_change = np.linalg.norm(x_solution - x_solution_old_iter)
        # Could also check norm of residual F_vector if desired
        print(f"[MIN-SURFACE] Iter {itr+1}/{max_iters} | ||Δx|| = {norm_change:.3e}")
        
        if norm_change < tolerance:
            print("[MIN-SURFACE] Newton's method converged.")
            break
    else: # Executed if loop finishes without break
        print("[MIN-SURFACE] Newton's method did not converge within max_iters.")

    # 1. Plot Jacobian J and Residual F (from the last iteration)
    if J_matrix is not None and F_vector is not None:
        matrix_plot_strings = plot_matrices_for_web(J_matrix, F_vector, title_suffix="(Minimal Surface - Final Iteration)")
        generated_plots.extend(matrix_plot_strings)
    else: # Should not happen if loop runs at least once
        print("[MIN-SURFACE] Jacobian/Residual not available for plotting.")

    # 2. Plot the 3D minimal surface
    fig_3d, ax_3d = plt.subplots(figsize=(9, 7), subplot_kw={'projection': '3d'})
    # skfem_plot3 might create its own figure if ax is not passed,
    # or might require specific setup. Check its signature.
    # Assuming skfem_plot3 can take an ax argument:
    try:
        skfem_plot3(mesh, x_solution, ax=ax_3d, cmap='viridis', edge_color='k', Nrefs=1) # Nrefs for smoother surface
        ax_3d.set_title("Minimal Surface Solution z(x,y)")
        ax_3d.set_xlabel("X-coordinate")
        ax_label_y = ax_3d.set_ylabel("Y-coordinate") # Store to adjust position if needed
        ax_3d.set_zlabel("Z-coordinate (height)")

        # Improve layout for 3D plot labels if they overlap
        # fig_3d.canvas.draw() # Ensure plot is drawn to get label positions
        # ax_label_y.set_position([0.9, 0.5]) # Example adjustment, may vary
        # ax_label_y.set_rotation(0)
        
        generated_plots.append(fig_to_base64_str(fig_3d))
    except Exception as e:
        print(f"Error during 3D plot: {e}")
        # Add a placeholder error plot for 3D view
        fig_err_3d, ax_err_3d = plt.subplots()
        ax_err_3d.text(0.5,0.5, "Error generating 3D plot.", ha='center', va='center')
        generated_plots.append(fig_to_base64_str(fig_err_3d))
        
    return mesh, basis, x_solution, generated_plots