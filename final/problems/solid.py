import numpy as np
import matplotlib.pyplot as plt
# matplotlib.tri is not explicitly used if plt.triplot is sufficient
from skfem import MeshTri, Basis, asm, solve, condense, FacetBasis
from skfem.element import ElementVector, ElementTriP2 # Using P2 vector elements
from skfem.helpers import sym_grad # For symmetric gradient
from skfem.visuals.matplotlib import draw as skfem_draw # Alias for clarity
from skfem import BilinearForm, LinearForm
# Import the new utilities
from utils.plot_utils import plot_matrices_for_web, fig_to_base64_str
from utils.mesh_utils import generate_random_stone_mesh


def solve_solid_for_web(mesh_type='rectangle', mesh_resolution_hint=None, E_val=None, nu_val=None, traction_val=1e6, deformed_scale=1e3):
    """
    Solve linear elasticity with traction on one edge.
    Returns displacement field and plots as base64 strings for web display.
    """
    generated_plots = []

    # --- Mesh selection ---
    if mesh_type == 'random':
        print(f"[SOLID] Random mesh type by generate_random_stone_mesh(). Res hint: {mesh_resolution_hint}")
        mesh = generate_random_stone_mesh()
    elif mesh_type == 'circle':
        print(f"[SOLID] Using built-in circular mesh. Res hint: {mesh_resolution_hint}")
        # Use skfem's built-in circle mesh
        nrefs = mesh_resolution_hint if mesh_resolution_hint is not None else 3
        mesh = MeshTri.init_circle(nrefs)
    elif mesh_type == 'rectangle':
        print(f"[SOLID] Using built-in rectangular mesh. Res hint: {mesh_resolution_hint}")
        # Use skfem's built-in square mesh (default is unit square)
        nrefs = mesh_resolution_hint if mesh_resolution_hint is not None else 3
        # Create a square mesh and refine it
        mesh = MeshTri().refined(nrefs)
    elif mesh_type == 'lshaped':
        print(f"[SOLID] Using L-shaped domain mesh. Res hint: {mesh_resolution_hint}")
        # Initialize L-shaped domain mesh
        mesh = MeshTri.init_lshaped()
        # Refine if needed
        if mesh_resolution_hint is not None and mesh_resolution_hint > 0:
            mesh = mesh.refined(mesh_resolution_hint)
    elif mesh_type == 'symmetric':
        print(f"[SOLID] Using symmetric square mesh. Res hint: {mesh_resolution_hint}")
        # Initialize the symmetric mesh of unit square
        mesh = MeshTri.init_symmetric()
        # Refine if needed
        if mesh_resolution_hint is not None and mesh_resolution_hint > 0:
            mesh = mesh.refined(mesh_resolution_hint)
    elif mesh_type == 'sqsymmetric':
        print(f"[SOLID] Using square symmetric mesh. Res hint: {mesh_resolution_hint}")
        # Initialize another variant of symmetric mesh
        mesh = MeshTri.init_sqsymmetric()
        # Refine if needed
        if mesh_resolution_hint is not None and mesh_resolution_hint > 0:
            mesh = mesh.refined(mesh_resolution_hint)
    elif mesh_type == 'tensor':
        print(f"[SOLID] Using tensor product mesh. Res hint: {mesh_resolution_hint}")
        # Determine the resolution based on the hint
        n_points = max(3, mesh_resolution_hint if mesh_resolution_hint is not None else 10)
        # Create a tensor product mesh with specified points in each dimension
        x = np.linspace(0, 1, n_points)
        y = np.linspace(0, 1, n_points)
        mesh = MeshTri.init_tensor(x, y)
    else:
        raise ValueError(f"Invalid mesh_type: {mesh_type}. Use 'circle', 'rectangle', 'lshaped', 'symmetric', 'sqsymmetric', 'tensor', or 'random'.")

    # Using P2 vector elements for displacement
    basis = Basis(mesh, ElementVector(ElementTriP2()), intorder=4) # Higher intorder for P2 elasticity

    # Material properties (Young's modulus E, Poisson's ratio nu)
    E_eff = E_val if E_val is not None else np.random.uniform(1e9, 5e10) # Pa
    nu_eff = nu_val if nu_val is not None else np.random.uniform(0.25, 0.33)
    
    # Lamé parameters
    lambda_lame = (E_eff * nu_eff) / ((1 + nu_eff) * (1 - 2 * nu_eff))
    mu_lame = E_eff / (2 * (1 + nu_eff))
    
    print(f"[SOLID] Effective Params: Mesh Vertices: {mesh.p.shape[1]} | E: {E_eff:.2e} Pa | ν: {nu_eff:.2f}")

    # Define bilinear form for linear elasticity (Kelvin-Voigt notation)
    @BilinearForm
    def elasticity_bilinear_form(u, v, w): # u: trial_vec, v: test_vec
        epsilon_u = sym_grad(u) # Symmetric gradient of trial function (strain tensor)
        epsilon_v = sym_grad(v) # Symmetric gradient of test function
        
        # Stress tensor sigma(u) = lambda_lame * tr(epsilon_u) * I + 2 * mu_lame * epsilon_u
        # Inner product: sum ( sigma(u)_ij * epsilon_v_ij )
        # = lambda * tr(eps_u)tr(eps_v) + 2*mu * sum(eps_u_ij * eps_v_ij) (for i,j)
        
        trace_epsilon_u = epsilon_u[0, 0] + epsilon_u[1, 1]
        trace_epsilon_v = epsilon_v[0, 0] + epsilon_v[1, 1]
        
        # Double contraction: eps_u : eps_v
        double_contraction = (epsilon_u[0, 0] * epsilon_v[0, 0] +
                              epsilon_u[1, 1] * epsilon_v[1, 1] +
                              2 * epsilon_u[0, 1] * epsilon_v[0, 1]) # Factor of 2 for shear components in engineering strain
                              
        return lambda_lame * trace_epsilon_u * trace_epsilon_v + 2 * mu_lame * double_contraction

    # Define linear form for traction boundary condition F_x = traction_val on right edge
    @LinearForm
    def traction_linear_form(v, w): # v: test_vec
        # Traction t = (traction_val, 0)
        # Integral (t . v) ds = Integral (traction_val * v[0]) ds
        return traction_val * v[0]

    # Assemble stiffness matrix K
    K_stiffness = asm(elasticity_bilinear_form, basis)

    # Identify facets on the "right" edge (e.g., x coordinate is maximal)
    # This assumes the mesh is somewhat aligned with axes.
    # For generate_random_stone_mesh, finding the "right" edge might be ambiguous.
    # A robust way is to find facets where x is close to mesh.p[0,:].max()
    mesh_x_max = mesh.p[0,:].max()
    right_edge_facets = mesh.facets_satisfying(lambda x_coords: np.isclose(x_coords[0], mesh_x_max))
    
    if not right_edge_facets.size > 0:
        print("[SOLID] Warning: No facets found on the 'right' edge (x=max(x)). Traction might not be applied.")

    # Create FacetBasis for applying traction
    traction_facet_basis = FacetBasis(mesh, basis.elem, facets=right_edge_facets)
    F_traction_load = asm(traction_linear_form, traction_facet_basis)
    
    # 1. Plot matrices K and F
    matrix_plot_strings = plot_matrices_for_web(K_stiffness, F_traction_load, title_suffix="(Solid Mechanics)")
    generated_plots.extend(matrix_plot_strings)

    # Define Dirichlet boundary conditions: Fix left edge (e.g., x coordinate is minimal)
    mesh_x_min = mesh.p[0,:].min()
    # Get DOFs on the "left" edge. For ElementVector, get_dofs returns DOFs for all components.
    left_edge_dofs = basis.get_dofs(mesh.facets_satisfying(lambda x_coords: np.isclose(x_coords[0], mesh_x_min))).flatten()
    
    if not left_edge_dofs.size > 0:
        print("[SOLID] Warning: No DOFs found on the 'left' edge (x=min(x)). Fixed BC might not be applied effectively.")

    # Solution vector for displacement U
    U_displacement = basis.zeros() # Initializes U with zeros, satisfying U=0 on left_edge_dofs initially.

    # Condense system (apply fixed BCs) and solve
    # K U = F. Fixed DOFs in left_edge_dofs are set to 0.
    # x0 specifies the values for Dirichlet DOFs (0 here).
    # U_displacement = solve(*condense(K_stiffness, F_traction_load, D=left_edge_dofs)) # This overwrites U_displacement
    
    # Manual condensation and expansion to be explicit:
    active_dofs = np.setdiff1d(np.arange(basis.N), left_edge_dofs)
    K_condensed, F_condensed = condense(K_stiffness, F_traction_load, D=left_edge_dofs, expand=False)

    try:
        U_free_dofs = solve(K_condensed, F_condensed)
        U_displacement[active_dofs] = U_free_dofs
        # U_displacement[left_edge_dofs] is already 0 from basis.zeros()
    except Exception as e:
        print(f"Error during solve: {e}")
        # Handle error, return with available plots
        return mesh, basis, U_displacement[basis.nodal_dofs], generated_plots


    # Extract nodal displacements for plotting
    # basis.nodal_dofs maps global DOF indices to [component, node_idx] type arrays for ElementVector
    # U_displacement[basis.nodal_dofs] should give (2, N_nodes) array for P2 elements
    nodal_displacement_vectors = U_displacement[basis.nodal_dofs]
    
    max_abs_disp = np.sqrt((nodal_displacement_vectors**2).sum(axis=0)).max()
    print(f"[SOLID] Max absolute displacement: {max_abs_disp:.3e} m")

    # 2. Displacement Field Quiver Plot
    fig_quiver, ax_quiver = plt.subplots(figsize=(8, 6))
    skfem_draw(mesh, ax=ax_quiver, M=mesh) # Draw original mesh
    ax_quiver.quiver(mesh.p[0, :], mesh.p[1, :],
                     nodal_displacement_vectors[0, :], nodal_displacement_vectors[1, :],
                     color='blue', angles='xy', scale_units='xy', scale=None) # Auto-scale
    ax_quiver.set_title("Displacement Field (U)")
    ax_quiver.set_xlabel("X-coordinate")
    ax_quiver.set_ylabel("Y-coordinate")
    ax_quiver.set_aspect('equal', adjustable='box')
    generated_plots.append(fig_to_base64_str(fig_quiver))

    # 3. Deformed vs. Original Mesh Plot
    fig_deformed, ax_deformed = plt.subplots(figsize=(8, 6))
    # Deformed mesh node positions: original_pos + scale_factor * displacement
    deformed_node_positions = mesh.p + deformed_scale * nodal_displacement_vectors
    
    ax_deformed.triplot(mesh.p[0, :], mesh.p[1, :], mesh.t.T, color='black', linestyle='--', linewidth=0.7, label=f'Original Mesh')
    ax_deformed.triplot(deformed_node_positions[0, :], deformed_node_positions[1, :], mesh.t.T, color='red', linewidth=0.9, label=f'Deformed Mesh (scale x{deformed_scale})')
    ax_deformed.legend()
    ax_deformed.set_title("Deformed vs. Original Mesh")
    ax_deformed.set_xlabel("X-coordinate")
    ax_deformed.set_ylabel("Y-coordinate")
    ax_deformed.set_aspect('equal', adjustable='box')
    generated_plots.append(fig_to_base64_str(fig_deformed))
    
    return mesh, basis, nodal_displacement_vectors, generated_plots