import numpy as np
import matplotlib.pyplot as plt
from skfem import MeshTri, Basis, asm, solve, condense, FacetBasis
from skfem.element import ElementVector, ElementTriP2
from skfem.helpers import sym_grad
from skfem.visuals.matplotlib import draw
from skfem import BilinearForm, LinearForm
from utils.plot_utils import plot_matrices
from utils.mesh_utils import generate_random_stone_mesh

def solve_solid(mesh_type='rectangle', mesh_resolution=None, E=None, nu=None):
    """
    Solve linear elasticity with traction on right edge.

    Parameters
    ----------
    mesh_type : str
        'rectangle', 'circle', or 'random'
    mesh_resolution : int
        Refinement level
    E : float
        Young's modulus
    nu : float
        Poisson's ratio
    """
    if mesh_type == 'random':
        mesh_type = np.random.choice(['circle', 'rectangle'])

    if mesh_type == 'circle':
        mesh_resolution = mesh_resolution or np.random.randint(3, 6)
        mesh = generate_random_stone_mesh()
    elif mesh_type == 'rectangle':
        mesh_resolution = mesh_resolution or np.random.randint(20, 40)
        mesh = generate_random_stone_mesh()
    else:
        raise ValueError("Invalid mesh_type")

    basis = Basis(mesh, ElementVector(ElementTriP2()))
    E = E or np.random.uniform(1e8, 1e11)
    nu = nu or np.random.uniform(0.25, 0.35)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    print(f"[SOLID] Mesh: {mesh_type} | Res: {mesh_resolution} | E: {E:.2e} | ν: {nu:.2f}")

    @BilinearForm
    def elasticity(u, v, w):
        eps_u = sym_grad(u)
        eps_v = sym_grad(v)
        trace_u = eps_u[0, 0] + eps_u[1, 1]
        trace_v = eps_v[0, 0] + eps_v[1, 1]
        return lam * trace_u * trace_v + 2 * mu * (
            eps_u[0, 0]*eps_v[0, 0] + eps_u[1, 1]*eps_v[1, 1] + 2 * eps_u[0, 1]*eps_v[0, 1]
        )

    @LinearForm
    def traction(v, w):
        return 1e6 * v[0]

    K = asm(elasticity, basis)
    right_facets = mesh.facets_satisfying(lambda x: np.isclose(x[0], 1.0))
    fbasis = FacetBasis(mesh, basis.elem, facets=right_facets)
    F = asm(traction, fbasis)
    plot_matrices(K, F, title_suffix="(Solid Mechanics)")

    left_dofs = basis.get_dofs(lambda x: np.isclose(x[0], 0.0)).flatten()
    x0 = np.zeros(basis.N)
    Kc, Fc = condense(K, F, D=left_dofs, x=x0, expand=False)
    U_free = solve(Kc, Fc)
    U = np.zeros(basis.N)
    free = np.setdiff1d(np.arange(basis.N), left_dofs)
    U[free] = U_free
    U[left_dofs] = x0[left_dofs]

    disp = U[basis.nodal_dofs]
    max_disp = np.sqrt((disp**2).sum(axis=0)).max()
    print(f"[SOLID] Max displacement: {max_disp:.6f} m")

    ax = draw(mesh)
    ax.quiver(*mesh.p, *disp, color='blue')
    ax.set_title("Displacement Field")
    plt.show()

    scale = 1e3
    def_p = mesh.p + scale * disp
    plt.figure()
    plt.triplot(mesh.p[0], mesh.p[1], mesh.t.T, 'k--', label='Original')
    plt.triplot(def_p[0], def_p[1], mesh.t.T, 'r-', label='Deformed')
    plt.legend()
    plt.title("Deformed vs Original Mesh")
    plt.show()

    return mesh, basis, disp
