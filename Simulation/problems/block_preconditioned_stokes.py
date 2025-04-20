from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt

from skfem import *
from skfem.models.poisson import vector_laplace, mass
from skfem.models.general import divergence
from skfem.assembly import LinearForm
from scipy.sparse import bmat, spmatrix
from scipy.sparse.linalg import LinearOperator, minres
from utils.plot_utils import plot_matrices


# --- Preconditioner setup ---
try:
    from pyamg import smoothed_aggregation_solver

    def build_pc(A: spmatrix, **kwargs) -> LinearOperator:
        return smoothed_aggregation_solver(A, **kwargs).aspreconditioner()
except ImportError:
    from scipy.sparse.linalg import spilu
    from scipy.sparse import csc_matrix

    def build_pc_ilu(A, **kwargs):
        ilu = spilu(csc_matrix(A), **kwargs)
        return lambda b: ilu.solve(b)

    def build_pc(A: spmatrix, **kwargs) -> LinearOperator:
        return build_pc_ilu(A, drop_tol=1e-3)


# --- Pressure L2 error form ---
class SphereDomain(NamedTuple):
    def pressure(self, x, y, z) -> np.ndarray:
        a, b, _ = np.ones(3)
        return (a**2 * (3 * a**2 + b**2) * x * y
                / (3 * a**4 + 2 * a**2 * b**2 + 3 * b**4))

    def pressure_error_form(self):
        def form(v, w):
            return v * (w['p'] - self.pressure(*w.x))
        return LinearForm(form)

def show_3d_mesh(mesh):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    p = mesh.p
    for tet in mesh.t.T:
        for i in range(4):
            for j in range(i + 1, 4):
                ax.plot(*p[:, [tet[i], tet[j]]], color='k', linewidth=0.5)

    ax.set_title("3D Tetrahedral Mesh")
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()

# --- Main solver ---
def solve_block_preconditioned_stokes(mesh_type='sphere', mesh_resolution=None):
    print(f"[STOKES-3D] Starting solver with mesh_type={mesh_type}")

    # --- Generate mesh ---
    if mesh_type == 'random':
        mesh_type = np.random.choice(['sphere', 'box'])

    if mesh_type == 'sphere':
        mesh = mesh = MeshTet.init_ball().refined(mesh_resolution or 2)

    elif mesh_type == 'box':
        from scipy.spatial import Delaunay
        pts = np.random.rand(100, 3)  # 500 random points in unit cube
        tets = Delaunay(pts)
        mesh = MeshTet(pts.T, tets.simplices.T)

    else:
        raise ValueError("Unsupported mesh_type. Use 'sphere', 'box', or 'random'.")

    print(f"[STOKES-3D] Mesh generated: {mesh.p.shape[1]} points, {mesh.t.shape[1]} tets")
    show_3d_mesh(mesh)

    domain = SphereDomain()

    # --- Basis and elements ---
    element = {
        'u': ElementVector(ElementTetP2()),
        'p': ElementTetP1()
    }

    basis = {
        var: Basis(mesh, e, intorder=3)
        for var, e in element.items()
    }

    @LinearForm
    def body_force(v, w):
        return w.x[0] * v[1]

    # --- Assemble matrices ---
    A = asm(vector_laplace, basis['u'])
    B = -asm(divergence, basis['u'], basis['p'])
    Q = asm(mass, basis['p'])

    K = bmat([[A, B.T],
              [B, None]], format='csr')

    f = np.concatenate([
        asm(body_force, basis['u']),
        basis['p'].zeros()
    ])

    # --- Condense BCs ---
    D = basis['u'].get_dofs()
    Kint, fint, u, I = condense(K, f, D=D)
    Aint = Kint[:-(basis['p'].N), :-(basis['p'].N)]

    # --- Visualize matrix ---
    plot_matrices(Kint, fint, title_suffix="(3D Stokes)")

    # --- Build preconditioner ---
    Apc = build_pc(Aint)
    diagQ = Q.diagonal()

    def precondition(uvp):
        uv, p = np.split(uvp, [Aint.shape[0]])
        return np.concatenate([Apc(uv), p / diagQ])

    M = LinearOperator(Kint.shape, matvec=precondition, dtype=Q.dtype)

    # --- Solve with Krylov MINRES ---
    from skfem.utils import solver_iter_krylov
    sol = solve(Kint, fint, u, I, solver=solver_iter_krylov(minres, verbose=True, M=M))

    # --- Split velocity and pressure ---
    velocity, pressure = np.split(sol, [basis['u'].N])

    # --- Pressure error ---
    error_p = asm(domain.pressure_error_form(), basis['p'],
                  p=basis['p'].interpolate(pressure))
    l2error_p = np.sqrt(error_p.T @ Q @ error_p)
    print(f"[STOKES-3D] L2 error in pressure: {l2error_p:.3e}")

    # --- Save VTK (optional) ---
    try:
        from pathlib import Path
        vtk_path = Path(__file__).with_suffix('.vtk')
        mesh.save(vtk_path, {
            'velocity': velocity[basis['u'].nodal_dofs].T,
            'pressure': pressure
        })
        print(f"[STOKES-3D] VTK file saved to: {vtk_path}")
    except Exception:
        print("[STOKES-3D] Skipped VTK export (optional)")

    return mesh, basis, velocity, pressure
