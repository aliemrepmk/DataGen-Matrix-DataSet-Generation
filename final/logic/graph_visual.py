"""logic.graph_visual
-------------------
Generate a colourful Network‑style rendering for a sparse matrix.
The helper chooses the **sample size adaptively** so we don’t accidentally
try to draw tens of thousands of nodes.  By default it takes 2 % of the
available vertices, bounded between 100 and 500 nodes, but you can pass a
custom `num_nodes`.

Dependencies: networkx ≥ 3.0 and graphviz command‑line (`fdp`).
"""

from __future__ import annotations

import os
import random
import tempfile
from typing import Optional

import numpy as np
import scipy.sparse as sp
from networkx import bfs_tree, from_scipy_sparse_array
from graphviz import Graph

__all__ = ["visualize_matrix_graph"]

# ──────────────────────────────────────────────────────────────────────────────

DEF_MIN_NODES = 340     # guarantee a meaningful blob
DEF_MAX_NODES = 1015     # keep layout time bounded
DEF_FRACTION  = 0.02    # 2 % of |V|


def _pick_sample_size(n_total: int,
                      num_nodes: Optional[int] = None) -> int:
    """Return a safe sample size <= n_total."""
    if num_nodes is not None and num_nodes > 0:
        return min(num_nodes, n_total)

    # adaptive choice:  max(DEF_MIN, 2 % of n, capped at DEF_MAX)
    est = int(DEF_FRACTION * n_total)
    return max(DEF_MIN_NODES, min(est, DEF_MAX_NODES, n_total))


def _bfs_sample(G, n_target: int, min_degree: int = 5,
                depth_limit: int = 10, seed: int = 42):
    """Return a subgraph with ≤ *n_target* nodes via BFS sampling."""
    rng = random.Random(seed)
    cand = [n for n, d in G.degree() if d >= min_degree] or list(G.nodes())
    start = rng.choice(cand)

    bfs_nodes = list(bfs_tree(G, start, depth_limit=depth_limit).nodes)[:n_target]
    return G.subgraph(bfs_nodes)


def visualize_matrix_graph(matrix: sp.spmatrix | np.ndarray,
                           save_path: str,
                           num_nodes: Optional[int] = None,
                           min_degree: int = 5,
                           depth_limit: int = 10,
                           seed: int = 42):
    """Create a PNG visualisation of *matrix* as a graph.

    Parameters
    ----------
    matrix      : sparse or dense square / rectangular array.
    save_path   : output PNG location.
    num_nodes   : desired node budget.  If *None*, a budget is chosen
                  via `_pick_sample_size`.
    min_degree  : only vertices with degree ≥ this value are considered
                  when picking a BFS root (falls back to any vertex).
    depth_limit : BFS depth bound.
    seed        : RNG seed for reproducible sampling.
    """

    if not sp.isspmatrix(matrix):
        matrix = sp.csr_matrix(matrix)

    n_total = matrix.shape[0]
    n_sample = _pick_sample_size(n_total, num_nodes)

    # build NetworkX graph once
    G_full = from_scipy_sparse_array(matrix)
    G_sub  = _bfs_sample(G_full, n_sample,
                         min_degree=min_degree,
                         depth_limit=depth_limit,
                         seed=seed)

    # build DOT graph
    dot = Graph(engine="fdp", strict=True)
    dot.attr(bgcolor="black")
    dot.attr("node", shape="point", color="red", width="0.05")
    dot.attr("edge", colorscheme="spectral11")

    rng = random.Random(seed)
    for u, v in G_sub.edges():
        dot.edge(str(u), str(v), color=str(rng.randint(1, 11)))

    # Write to PNG through a temp .dot to avoid pipe issues on Windows
    with tempfile.TemporaryDirectory() as tmp:
        dot_file = os.path.join(tmp, "tmp.dot")
        dot.save(dot_file)
        os.system(f"fdp -Tpng {dot_file} -o {save_path}")

    return save_path
