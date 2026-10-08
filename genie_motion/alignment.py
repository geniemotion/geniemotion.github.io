import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def umeyama_alignment(X, Y):
    """Similarity transform (scale, rotation, translation) mapping X onto Y."""
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)

    Xc = X - mu_x
    Yc = Y - mu_y

    Sigma = (Yc.T @ Xc) / X.shape[0]
    U, D, Vt = np.linalg.svd(Sigma)

    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    var_x = np.sum(Xc**2) / X.shape[0]

    if var_x < 1e-9 or not np.isfinite(var_x):
        s = 1.0   # degenerate spread: skip scale correction, keep rotation + translation
    else:
        s = np.sum(D) / var_x
        if not np.isfinite(s) or s <= 0:
            s = 1.0

    t = mu_y - s * R @ mu_x

    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def global_ransac(src_pts, tgt_pts, voxel_size):
    """Global registration from FPFH feature matches (used when ICP finds no correspondences)."""
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_pts)

    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(tgt_pts)

    src_down = src.voxel_down_sample(voxel_size)
    tgt_down = tgt.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    src_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    tgt_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down,
        src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 2,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 2)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(40000, 500)
    )

    return result.transformation


def icp_o3d(src_pts, tgt_pts, init_T, OUT_DIR=""):
    """
    Coarse-to-fine point-to-plane ICP from `init_T`, followed by a similarity
    refinement. Falls back to feature-based RANSAC if ICP finds too few
    correspondences. Returns (T, rmse, registration result).
    """
    src_pts = np.asarray(src_pts)
    tgt_pts = np.asarray(tgt_pts)

    # guard: empty or near-empty point clouds
    if len(src_pts) < 3 or len(tgt_pts) < 3:
        print(
            f"ICP skipped: too few points "
            f"(src={len(src_pts)}, tgt={len(tgt_pts)})"
        )
        return init_T, np.inf, None

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_pts)

    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(tgt_pts)

    src_np = np.asarray(src.points)
    tgt_np = np.asarray(tgt.points)

    bbox = np.linalg.norm(tgt_np.max(0) - tgt_np.min(0))
    radius = 0.05 * bbox

    src.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    tgt.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )

    tree = cKDTree(tgt_np)
    dists, _ = tree.query(src_np, k=1)

    median_dist = np.median(dists)
    if median_dist == 0 or np.isnan(median_dist):
        median_dist = 0.01 * bbox

    # hard floor so Open3D never sees a non-positive threshold
    MIN_CORR_DIST = 1e-6
    if median_dist < MIN_CORR_DIST:
        print(f"ICP: median_dist={median_dist} too small, clamping to {MIN_CORR_DIST}")
        median_dist = MIN_CORR_DIST

    thresholds = [5*median_dist, 2*median_dist, 1*median_dist]
    T = init_T.copy()

    for th in thresholds:
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, th, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        T = reg.transformation

    num_corr = len(reg.correspondence_set)
    rmse = reg.inlier_rmse

    print(f"[ICP] corr: {num_corr}, rmse: {rmse:.4f}")

    if num_corr < 20:
        print("ICP failed, running RANSAC")

        voxel = 0.05 * bbox
        T_ransac = global_ransac(src_pts, tgt_pts, voxel)

        # refine with ICP
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt,
            max_correspondence_distance=2 * median_dist,
            init=T_ransac,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        print(f"[RANSAC] recovered rmse: {reg.inlier_rmse:.4f}")

        return reg.transformation, reg.inlier_rmse, reg

    if rmse > 5 * median_dist:
        print("Bad alignment, rejecting")
        return init_T, rmse, reg

    src_trans = (T[:3, :3] @ src_np.T).T + T[:3, 3]

    src_corr, tgt_corr = [], []
    for p in src_trans:
        dist, idx = tree.query(p, k=1)
        if dist < 2 * median_dist:
            src_corr.append(p)
            tgt_corr.append(tgt_np[idx])

    if len(src_corr) > 30:
        T_sim3 = umeyama_alignment(np.array(src_corr), np.array(tgt_corr))
        if len(OUT_DIR) > 0:
            np.save(os.path.join(OUT_DIR, "T_sim3.npy"), T_sim3)
        T = T_sim3 @ T

    return T, rmse, reg


def pca_axes(pts):
    pts_c = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts_c)
    return Vt


def pca_init(src, tgt):
    """Initial rigid transform aligning the principal axes and centroids of src to tgt."""
    R_src = pca_axes(src)
    R_tgt = pca_axes(tgt)

    R = R_tgt.T @ R_src

    t = tgt.mean(axis=0) - R @ src.mean(axis=0)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T
