import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.sparse.linalg import splu
from skfem import MeshTri, Basis, asm, solve, condense, enforce, LinearForm, BilinearForm
from skfem.element import ElementTriP1
from skfem.helpers import grad, dot
from skfem.visuals.matplotlib import draw
from utils.plot_utils import plot_matrices
from utils.mesh_utils import generate_random_stone_mesh


def solve_heat(mesh_type='rectangle', mesh_resolution=None, kappa=None):
    """
    Solve transient heat conduction using Crank-Nicolson scheme.

    Parameters
    ----------
    mesh_type : str
        'rectangle', 'circle', or 'random'.
    mesh_resolution : int
        Mesh refinement level or grid resolution.
    kappa : float
        Thermal diffusivity (if None, randomized in [0.1, 5.0]).
    """
    # --- Mesh selection ---
    if mesh_type == 'random':
        mesh_type = np.random.choice(['circle', 'rectangle'])

    if mesh_type == 'circle':
        mesh_resolution = mesh_resolution or np.random.randint(3, 6)
        mesh = generate_random_stone_mesh()
    elif mesh_type == 'rectangle':
        mesh_resolution = mesh_resolution or np.random.randint(20, 40)
        mesh = generate_random_stone_mesh()
    else:
        raise ValueError("Invalid mesh_type. Use 'circle', 'rectangle', or 'random'.")

    basis = Basis(mesh, ElementTriP1())

    @BilinearForm
    def laplace(u, v, w):
        return dot(grad(u), grad(v))

    @BilinearForm
    def mass(u, v, w):
        return u * v

    K = asm(laplace, basis)
    M = asm(mass, basis)

    D = basis.get_dofs()
    K_bc = enforce(K, D=D)
    M_bc = enforce(M, D=D)

    # --- Time stepping ---
    t_end = 0.5
    dt = 0.01
    theta = 0.7
    n_steps = int(t_end / dt)

    # --- Initial Condition ---
    X, Y = basis.doflocs
    T = np.cos(np.pi * X) * np.cos(np.pi * Y)
    T[D] = 0.0

    # --- Thermal diffusivity ---
    kappa = kappa or np.random.uniform(0.1, 5.0)
    print(f"[HEAT] Mesh: {mesh_type} | Res: {mesh_resolution} | κ = {kappa:.2f}")

    A_time = M_bc + theta * dt * kappa * K_bc
    B_time = M_bc - (1 - theta) * dt * kappa * K_bc
    plot_matrices(A_time, B_time @ T, title_suffix="(Heat Transfer)")

    A_factor = splu(A_time.T.tocsc())
    for _ in range(n_steps):
        RHS = B_time @ T
        T = A_factor.solve(RHS)
        T[D] = 0.0

    print(f"[HEAT] Max temperature at t={t_end}: {T.max():.3f}")

    tri = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    plt.figure()
    plt.tricontourf(tri, T, cmap='hot')
    plt.title("Heat Transfer: Temperature Field")
    plt.colorbar()
    plt.show()

    return mesh, basis, T
