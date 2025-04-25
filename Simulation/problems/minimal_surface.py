import numpy as np
import matplotlib.pyplot as plt
from skfem import MeshTri, Basis, asm, solve, condense, LinearForm, BilinearForm
from skfem.helpers import grad, dot
from skfem.element import ElementTriP1
from skfem.visuals.matplotlib import plot3, show
from utils.plot_utils import plot_matrices
from utils.mesh_utils import generate_random_stone_mesh


def solve_minimal_surface(mesh_type='rectangle', mesh_resolution=None):
    """
    Solve the nonlinear minimal surface equation using Newton's method.

    Parameters
    ----------
    mesh_type : str
        'rectangle', 'circle', or 'random'
    mesh_resolution : int or None
        Mesh refinement level (ignored since mesh is generated via point clouds)
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

    basis = Basis(mesh, ElementTriP1())
    x = basis.zeros()
    D = mesh.boundary_nodes()
    x[D] = np.sin(np.pi * mesh.p[0, D])

    print(f"[MIN-SURFACE] Mesh: {mesh_type} | Resolution: {mesh_resolution}")

    @BilinearForm
    def jacobian(u, v, w):
        w_val = w['prev']
        g = grad(w_val)
        norm_g = np.sqrt(1 + dot(g, g))
        return (dot(grad(u), grad(v)) / norm_g
                - dot(grad(u), grad(w_val)) * dot(grad(w_val), grad(v)) / norm_g**3)

    @LinearForm
    def rhs(v, w):
        w_val = w['prev']
        return dot(grad(w_val), grad(v)) / np.sqrt(1 + dot(grad(w_val), grad(w_val)))

    for itr in range(100):
        w_interp = basis.interpolate(x)
        J = asm(jacobian, basis, prev=w_interp)
        F = asm(rhs, basis, prev=w_interp)
        x_prev = x.copy()
        x += 0.7 * solve(*condense(J, -F, D=D))
        norm_change = np.linalg.norm(x - x_prev)
        print(f"[MIN-SURFACE] Iter {itr} | Δx = {norm_change:.2e}")
        if norm_change < 1e-8:
            print("[MIN-SURFACE] Converged.")
            break

    plot_matrices(J, F, title_suffix="(Minimal Surface)")
    plot3(mesh, x)
    show()

    return mesh, basis, x
