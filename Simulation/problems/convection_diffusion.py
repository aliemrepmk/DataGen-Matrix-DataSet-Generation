import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from skfem import MeshTri, Basis, asm, solve, condense, LinearForm, BilinearForm
from skfem.element import ElementTriP1
from skfem.helpers import grad, dot
from skfem.visuals.matplotlib import draw
from utils.plot_utils import plot_matrices
from utils.mesh_utils import generate_random_stone_mesh


def solve_convection_diffusion_problem(mesh_type="rectangle", mesh_res=None,
                                       epsilon=None, b_field=None, f_value=None):
    """
    Solve -ε Δu + b·∇u = f with Dirichlet BCs using FEM.

    Parameters
    ----------
    mesh_type : str
        'rectangle', 'circle', or 'random'
    mesh_res : int
        Mesh refinement level
    epsilon : float
        Diffusion coefficient
    b_field : tuple
        Advection vector (bx, by)
    f_value : float
        Source term
    """
    if mesh_type == 'random':
        mesh_type = np.random.choice(['circle', 'rectangle'])

    if mesh_type == 'circle':
        mesh_res = mesh_res or np.random.randint(3, 6)
        mesh = generate_random_stone_mesh()
    elif mesh_type == 'rectangle':
        mesh_res = mesh_res or np.random.randint(20, 40)
        mesh = generate_random_stone_mesh()
    else:
        raise ValueError("Invalid mesh_type.")

    basis = Basis(mesh, ElementTriP1(), intorder=3)

    epsilon = epsilon or np.random.uniform(1e-4, 1e-2)
    b_field = b_field or (np.random.uniform(0.5, 2.0), np.random.uniform(-0.5, 0.5))
    f_value = f_value or np.random.uniform(0.5, 5.0)
    bx, by = b_field

    print(f"[CONV-DIFF] Mesh: {mesh_type} | Res: {mesh_res} | ε: {epsilon:.2e} | b: {b_field} | f: {f_value:.2f}")

    @BilinearForm
    def diffusion(u, v, w):
        return epsilon * dot(grad(u), grad(v))

    @BilinearForm
    def advection(u, v, w):
        return (bx * grad(u)[0] + by * grad(u)[1]) * v

    A = asm(diffusion, basis) + asm(advection, basis)

    @LinearForm
    def rhs(v, w):
        return f_value * v

    F = asm(rhs, basis)
    D = mesh.boundary_nodes()
    A_cond, F_cond = condense(A, F, D=D, expand=False)
    sol = solve(A_cond, F_cond)

    u_full = np.zeros(basis.N)
    u_full[np.setdiff1d(np.arange(basis.N), D)] = sol

    plot_matrices(A, F, title_suffix="(Convection–Diffusion)")

    tri = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    plt.figure()
    plt.tricontourf(tri, u_full, cmap='hot')
    plt.title("Convection–Diffusion: Solution")
    plt.colorbar()
    plt.show()

    return mesh, basis, u_full
