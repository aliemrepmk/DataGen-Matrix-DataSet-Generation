from skfem import (MeshTri, Basis, asm, solve, condense, bmat, LinearForm,
                   BilinearForm) # Added BilinearForm for non-symmetric case
from skfem.element import ElementVector, ElementTriP2, ElementTriP1
from skfem.models.poisson import vector_laplace, mass
from skfem.models.general import divergence, rot
# Import skfem visual tools with aliases to avoid clashes if needed
from skfem.visuals.matplotlib import draw as skfem_draw
from skfem.visuals.matplotlib import plot as skfem_plot
from utils.mesh_utils import generate_random_stone_mesh
# Import the new utilities
from utils.plot_utils import plot_matrices_for_web, fig_to_base64_str
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.sparse import csr_matrix # For non-symmetric case
from skfem.helpers import grad, dot, sym_grad # Add or ensure 'grad' is here

def solve_fluid_symmetric_custom_for_web(mesh_type, mesh_resolution_hint, viscosity, reg_param):
    """
    Solves a customizable symmetric fluid mechanics problem (Stokes flow).
    Returns K, f, solution fields, and plots as base64 strings for web display.
    mesh_resolution_hint is a general hint for mesh resolution.
    """
    generated_plots = []

    print(f"[FLUID-SYM-CUSTOM] Mesh type: {mesh_type}, Visc: {viscosity:.2f}, Reg: {reg_param:.2e}")
    
    # --- Mesh selection ---
    if mesh_type == 'random':
        print(f"[FLUID] Random mesh using generate_random_stone_mesh")
        mesh = generate_random_stone_mesh()
    elif mesh_type == 'circle':
        print(f"[FLUID] Using built-in circular mesh. Res hint: {mesh_resolution_hint}")
        # Use skfem's built-in circle mesh
        nrefs = mesh_resolution_hint if mesh_resolution_hint is not None else 3
        mesh = MeshTri.init_circle(nrefs)
    elif mesh_type == 'rectangle':
        print(f"[FLUID] Using built-in rectangular mesh. Res hint: {mesh_resolution_hint}")
        # Use skfem's built-in square mesh (default is unit square)
        nrefs = mesh_resolution_hint if mesh_resolution_hint is not None else 3
        # Create a square mesh and refine it
        mesh = MeshTri().refined(nrefs)
    elif mesh_type == 'lshaped':
        print(f"[FLUID] Using L-shaped domain mesh. Res hint: {mesh_resolution_hint}")
        # Initialize L-shaped domain mesh
        mesh = MeshTri.init_lshaped()
        # Refine if needed
        if mesh_resolution_hint is not None and mesh_resolution_hint > 0:
            mesh = mesh.refined(mesh_resolution_hint)
    elif mesh_type == 'symmetric':
        print(f"[FLUID] Using symmetric square mesh. Res hint: {mesh_resolution_hint}")
        # Initialize the symmetric mesh of unit square
        mesh = MeshTri.init_symmetric()
        # Refine if needed
        if mesh_resolution_hint is not None and mesh_resolution_hint > 0:
            mesh = mesh.refined(mesh_resolution_hint)
    elif mesh_type == 'sqsymmetric':
        print(f"[FLUID] Using square symmetric mesh. Res hint: {mesh_resolution_hint}")
        # Initialize another variant of symmetric mesh
        mesh = MeshTri.init_sqsymmetric()
        # Refine if needed
        if mesh_resolution_hint is not None and mesh_resolution_hint > 0:
            mesh = mesh.refined(mesh_resolution_hint)
    elif mesh_type == 'tensor':
        print(f"[FLUID] Using tensor product mesh. Res hint: {mesh_resolution_hint}")
        # Determine the resolution based on the hint
        n_points = max(3, mesh_resolution_hint if mesh_resolution_hint is not None else 10)
        # Create a tensor product mesh with specified points in each dimension
        x = np.linspace(0, 1, n_points)
        y = np.linspace(0, 1, n_points)
        mesh = MeshTri.init_tensor(x, y)
    else:
        print(f"[FLUID] Unknown mesh type '{mesh_type}', defaulting to random")
        mesh = generate_random_stone_mesh()
    
    print(f"[FLUID-SYM-CUSTOM] Generated mesh with {mesh.p.shape[1]} vertices.")

    element = {
        'u': ElementVector(ElementTriP2()),
        'p': ElementTriP1()
    }
    basis = {var: Basis(mesh, elem, intorder=4) for var, elem in element.items()}

    A = asm(vector_laplace, basis['u']) * viscosity
    B_mat = asm(divergence, basis['u'], basis['p'])
    C_mass = asm(mass, basis['p'])
    K = bmat([[A, -B_mat.T],
              [-B_mat, reg_param * C_mass]], format='csr')

    @LinearForm
    def body_force(v, w):
        return w.x[0] * v[1]

    f_u = asm(body_force, basis['u'])
    f_p = basis['p'].zeros()
    f_vec = np.concatenate([f_u, f_p])

    # 1. Plot matrices
    matrix_plot_strings = plot_matrices_for_web(K, f_vec, title_suffix="(Symmetric Custom Fluid)")
    generated_plots.extend(matrix_plot_strings)
    
    boundary_dofs_u = basis['u'].get_dofs(mesh.boundary_nodes())
    uvp = solve(*condense(K, f_vec, D=boundary_dofs_u))
    velocity, pressure = np.split(uvp, [A.shape[0]])

    basis['psi'] = Basis(mesh, ElementTriP2(), intorder=4)
    A_psi = asm(vector_laplace, basis['psi'])
    velocity_interpolated = basis['u'].interpolate(velocity)
    vorticity_form_val = asm(rot, basis['psi'], w=velocity_interpolated)
    boundary_dofs_psi = basis['psi'].get_dofs(mesh.boundary_nodes())
    psi = solve(*condense(A_psi, vorticity_form_val, D=boundary_dofs_psi))

    # 2. Velocity Field Plot
    fig_vel, ax_vel = plt.subplots(figsize=(7, 7))
    skfem_draw(mesh, ax=ax_vel, M=mesh)
    u_nodes = velocity[basis['u'].nodal_dofs]
    ax_vel.quiver(mesh.p[0, :], mesh.p[1, :], u_nodes[0, :], u_nodes[1, :], color='r')
    ax_vel.set_title(f"Velocity (Custom V={viscosity:.1f}, R={reg_param:.1e})")
    ax_vel.set_xlabel("X"); ax_vel.set_ylabel("Y"); ax_vel.set_aspect('equal', adjustable='box')
    generated_plots.append(fig_to_base64_str(fig_vel))

    # 3. Pressure Field Plot
    fig_press, ax_press = plt.subplots(figsize=(7, 6))
    skfem_plot(basis['p'], pressure, ax=ax_press, shading='gouraud', M=mesh, Nrefs=2)
    fig_press.colorbar(ax_press.get_children()[0], ax=ax_press, label="Pressure")
    ax_press.set_title("Pressure Field (Custom)")
    ax_press.set_xlabel("X"); ax_press.set_ylabel("Y"); ax_press.set_aspect('equal', adjustable='box')
    generated_plots.append(fig_to_base64_str(fig_press))

    # 4. Stream-Function Contours Plot
    fig_stream, ax_stream = plt.subplots(figsize=(7, 6))
    tri_viz = mtri.Triangulation(mesh.p[0, :], mesh.p[1, :], mesh.t.T)
    psi_nodal_values = psi[basis['psi'].nodal_dofs.flatten()]
    contour = ax_stream.tricontourf(tri_viz, psi_nodal_values, cmap='viridis', levels=20)
    fig_stream.colorbar(contour, ax=ax_stream, label="$\psi$")
    ax_stream.set_title("Stream-Function (Custom)")
    ax_stream.set_xlabel("X"); ax_stream.set_ylabel("Y"); ax_stream.set_aspect('equal', adjustable='box')
    generated_plots.append(fig_to_base64_str(fig_stream))

    return K, f_vec, mesh, basis, velocity, pressure, psi, generated_plots
