import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict


def build_vertex_correspondence(vertices, depth_points):
    """
    Parameters
    ----------
    vertices : (M,3) ndarray
        Current transformed mesh vertices.

    depth_points : (N,3) ndarray
        Current depth point cloud.

    Returns
    -------
    vertex_to_depth : defaultdict(list)
        vertex_id -> [(depth_id, distance), ...]

    depth_to_vertex : ndarray
        depth_id -> vertex_id
    """

    tree = cKDTree(vertices)

    distances, vertex_ids = tree.query(depth_points)

    vertex_to_depth = defaultdict(list)

    for depth_id, (vertex_id, dist) in enumerate(zip(vertex_ids, distances)):
        vertex_to_depth[int(vertex_id)].append(
            (depth_id, float(dist))
        )

    return vertex_to_depth, vertex_ids