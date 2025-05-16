import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from skfem import MeshTri
import matplotlib.pyplot as plt

def generate_random_stone_mesh():
    np.random.seed()
    mode = np.random.choice([
        'clustered', 'convex_blob', 'noisy_rectangle',
        'thin_plate', 'triangle_cloud', 'chaotic_combo'
    ])
    points = []

    if mode == 'clustered':
        n_clusters = np.random.randint(2, 8)
        for _ in range(n_clusters):
            center = np.random.uniform(-2, 2, 2)
            spread = np.random.uniform(0.2, 1.0)
            n_points = np.random.randint(20, 100)
            cluster = np.random.randn(n_points, 2) * spread + center
            points.append(cluster)

    elif mode == 'convex_blob':
        base = np.random.rand(50, 2) * np.random.uniform(2, 5)
        hull = ConvexHull(base)
        points = base[hull.vertices]
        noise = np.random.randn(200, 2) * 0.3
        points = np.vstack([points, noise + np.mean(points, axis=0)])

    elif mode == 'noisy_rectangle':
        w, h = np.random.uniform(2, 4), np.random.uniform(1, 2)
        x = np.random.rand(300) * w - w / 2
        y = np.random.rand(300) * h - h / 2
        points = np.column_stack([x, y]) + np.random.randn(300, 2) * 0.1

    elif mode == 'thin_plate':
        length = np.random.uniform(4, 8)
        width = np.random.uniform(0.2, 0.5)
        x = (np.random.rand(300) - 0.5) * length
        y = (np.random.rand(300) - 0.5) * width
        points = np.column_stack([x, y]) + np.random.randn(300, 2) * 0.05

    elif mode == 'triangle_cloud':
        corners = np.array([[0, 0], [2, 0], [1, 2]]) * np.random.uniform(0.8, 1.5)
        points = []
        for _ in range(300):
            a, b, c = np.random.dirichlet([1, 1, 1])
            p = a * corners[0] + b * corners[1] + c * corners[2]
            points.append(p + np.random.randn(2) * 0.05)
        points = np.array(points)

    elif mode == 'chaotic_combo':
        points = []
        for _ in range(np.random.randint(3, 7)):
            center = np.random.uniform(-2, 2, 2)
            cluster = np.random.randn(np.random.randint(30, 100), 2) * np.random.uniform(0.2, 0.7) + center
            points.append(cluster)
        points = np.vstack(points)
        global_noise = np.random.uniform(-3, 3, size=(100, 2))
        points = np.vstack([points, global_noise])

    points = np.vstack(points)
    tri = Delaunay(points)
    mesh = MeshTri(points.T, tri.simplices.T)
    mesh.draw()
    plt.title(f"Random Mesh Mode: {mode}")
    plt.axis("equal")
    plt.show()

    return mesh
