from skfem import MeshTri, Basis, asm, solve, condense, bmat, LinearForm
from skfem.element import ElementVector, ElementTriP2, ElementTriP1
from skfem.models.poisson import vector_laplace, mass
from skfem.models.general import divergence, rot
from skfem.visuals.matplotlib import draw, plot
from utils.mesh_utils import generate_random_stone_mesh
from utils.plot_utils import plot_matrices
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def solve_fluid_symmetric():
    mesh = MeshTri.init_circle(4)
    element = {
        'u': ElementVector(ElementTriP2()),
        'p': ElementTriP1()
    }
    basis = {var: Basis(mesh, elem, intorder=3) for var, elem in element.items()}
    A = asm(vector_laplace, basis['u']) * 10.0
    B_mat = asm(divergence, basis['u'], basis['p'])
    C = asm(mass, basis['p'])
    K = bmat([[A, -B_mat.T],
              [-B_mat, 3e-6 * C]], format='csr')

    @LinearForm
    def body_force(v, w):
        return w.x[0] * v[1]

    f_u = asm(body_force, basis['u'])
    f_p = basis['p'].zeros()
    f = np.concatenate([f_u, f_p])
    plot_matrices(K, f, title_suffix="(Symmetric)")
    uvp = solve(*condense(K, f, D=basis['u'].get_dofs()))
    velocity, pressure = np.split(uvp, [A.shape[0]])
    basis['psi'] = basis['u'].with_element(ElementTriP2())
    A_psi = asm(vector_laplace, basis['psi'])
    vorticity = asm(rot, basis['psi'], w=basis['u'].interpolate(velocity))
    psi = solve(*condense(A_psi, vorticity, D=basis['psi'].get_dofs()))

    vel_nodal = velocity[basis['u'].nodal_dofs]
    ax = draw(mesh)
    ax.quiver(*mesh.p, *vel_nodal, color='r')
    ax.set_title("Fluid Mechanics: Velocity Field (Symmetric)")
    plt.show()

    plt.figure()
    plot(basis['p'], pressure)
    plt.title("Pressure Field")
    plt.show()

    plt.figure()
    tri = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    plt.tricontour(tri, psi[basis['psi'].nodal_dofs.flatten()], cmap='viridis')
    plt.colorbar()
    plt.title("Stream-Function Contours")
    plt.show()

    return mesh, basis, velocity, pressure, psi


def solve_fluid_symmetric_custom(mesh_type, mesh_resolution, viscosity, reg_param):
    if mesh_type == "circle":
        mesh = generate_random_stone_mesh()
    elif mesh_type == "rectangle":
        mesh = generate_random_stone_mesh()
    else:
        raise ValueError("Unknown mesh type")

    element = {
        'u': ElementVector(ElementTriP2()),
        'p': ElementTriP1()
    }
    basis = {var: Basis(mesh, elem, intorder=3) for var, elem in element.items()}
    A = asm(vector_laplace, basis['u']) * viscosity
    B_mat = asm(divergence, basis['u'], basis['p'])
    C = asm(mass, basis['p'])
    K = bmat([[A, -B_mat.T],
              [-B_mat, reg_param * C]], format='csr')

    @LinearForm
    def body_force(v, w):
        return w.x[0] * v[1]

    f_u = asm(body_force, basis['u'])
    f_p = basis['p'].zeros()
    f = np.concatenate([f_u, f_p])
    plot_matrices(K, f, title_suffix="(Symmetric Custom)")
    uvp = solve(*condense(K, f, D=basis['u'].get_dofs()))
    velocity, pressure = np.split(uvp, [A.shape[0]])
    basis['psi'] = basis['u'].with_element(ElementTriP2())
    A_psi = asm(vector_laplace, basis['psi'])
    vorticity = asm(rot, basis['psi'], w=basis['u'].interpolate(velocity))
    psi = solve(*condense(A_psi, vorticity, D=basis['psi'].get_dofs()))

    return K, f, mesh, basis, velocity, pressure, psi


def solve_fluid_nonsymmetric():
    mesh = MeshTri.init_circle(4)
    element = {
        'u': ElementVector(ElementTriP2()),
        'p': ElementTriP1()
    }
    basis = {var: Basis(mesh, elem, intorder=3) for var, elem in element.items()}
    A = asm(vector_laplace, basis['u']) * 10.0
    dof_coords = basis['u'].doflocs
    conv_field = (100.0 * dof_coords[0], 50.0 * dof_coords[1])
    w_field = np.vstack(conv_field)

    from skfem import BilinearForm

    @BilinearForm
    def convection(u, v, w):
        return w.x[0] * u[1] * v[0]

    conv_term = asm(convection, basis['u'], w=w_field)
    A_total = A + conv_term
    from scipy.sparse import csr_matrix
    n = A_total.shape[0]
    perm = np.random.permutation(n)
    P = csr_matrix((np.ones(n), (perm, np.arange(n))), shape=(n, n))
    A_total = P @ A_total @ P.T
    B_mat = asm(divergence, basis['u'], basis['p'])
    C = asm(mass, basis['p'])
    K = bmat([[A_total, -B_mat.T],
              [-B_mat, 3e-6 * C]], format='csr')

    @LinearForm
    def body_force(v, w):
        return w.x[0] * v[1]

    f_u = asm(body_force, basis['u'])
    f_p = basis['p'].zeros()
    f = np.concatenate([f_u, f_p])
    plot_matrices(K, f, title_suffix="(Non-Symmetric)")
    uvp = solve(*condense(K, f, D=basis['u'].get_dofs()))
    velocity, pressure = np.split(uvp, [A.shape[0]])
    basis['psi'] = basis['u'].with_element(ElementTriP2())
    A_psi = asm(vector_laplace, basis['psi'])
    vorticity = asm(rot, basis['psi'], w=basis['u'].interpolate(velocity))
    psi = solve(*condense(A_psi, vorticity, D=basis['psi'].get_dofs()))

    vel_nodal = velocity[basis['u'].nodal_dofs]
    ax = draw(mesh)
    ax.quiver(*mesh.p, *vel_nodal, color='r')
    ax.set_title("Velocity Field (Non-Symmetric)")
    plt.show()

    plt.figure()
    plot(basis['p'], pressure)
    plt.title("Pressure Field")
    plt.show()

    plt.figure()
    tri = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    plt.tricontour(tri, psi[basis['psi'].nodal_dofs.flatten()], cmap='viridis')
    plt.colorbar()
    plt.title("Stream-Function Contours")
    plt.show()

    return mesh, basis, velocity, pressure, psi
