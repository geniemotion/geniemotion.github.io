"""
GenieMotion: canonical-geometry-grounded SE(3) trajectory recovery.

Runs the final stage of the pipeline on a sample whose preprocessing (canonical
mesh reconstruction, SAM2 tracking, depth alignment) is already done:

    1. register the canonical mesh G to the first depth cloud, then propagate it
       through the rollout with point-to-plane ICP;
    2. fit one transportation map per vertex (rigid prior + RBF-GP residual)
       against the depth observations;
    3. project the corrected vertices back onto SE(3) . G with Kabsch;
    4. compose with the grasp transform to get the end-effector trajectory.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import shutup
import tqdm
from PIL import Image

from affine_transform import AffineWarper
from alignment import icp_o3d, pca_init
from get_trajectory import kabsch
from utils import (
    depth_to_pts,
    fit_canonical_to_bbox,
    load_canonical,
    prepare_depth,
    transform_points,
)
from vertex_correspondence import build_vertex_correspondence
from visualize import visualize_vertex_flow_on_image, visualize_vertex_trajectories_on_image
from warping_transform_gpu import BatchGPWarper


def main():
    shutup.please()

    parser = argparse.ArgumentParser()
    parser.add_argument("--take-dir", required=True,
                        help="Folder holding initial_rgb.png, initial_depth.npy and objs<suffix>/.")
    parser.add_argument("--sample", required=True,
                        help="Sub-folder of --take-dir holding aligned_depths.npz and tracking_template<suffix>/.")
    parser.add_argument("--suffix", default="_sam2",
                        help="Reads objs<suffix>/ and tracking_template<suffix>/, and writes "
                             "trajectory_ee<suffix>_allGP.npy, actionable_flow<suffix>_allGP.png and "
                             "vertex_flow_2d<suffix>_allGP.png into the sample folder.")
    parser.add_argument("--camera-calibration", default=None,
                        help="Optional camera calibration JSON ({camera: {'intrinsics': 3x3}}). "
                             "If omitted, pinhole intrinsics with f = 0.8 * max(H, W) and the "
                             "principal point at the image centre are used.")
    parser.add_argument("--camera-name", default=None,
                        help="Camera entry inside the calibration JSON. "
                             "If omitted, the first entry is used.")
    parser.add_argument("--depth-dtype", choices=["float32", "uint16"], default="uint16",
                        help="Depth dtype after scale/shift alignment: uint16 for raw RealSense "
                             "depth in mm (default), float32 for simulators.")
    args = parser.parse_args()

    suffix = args.suffix
    output_suffix = suffix + "_allGP"

    sample_dir = os.path.join(args.take_dir, args.sample)
    out_dir = sample_dir + "/"
    rgb_path = os.path.join(args.take_dir, "initial_rgb.png")
    max_gp_vertices = np.inf  # cap on the number of vertices with their own GP

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    with open(os.path.join(sample_dir, f"tracking_template{suffix}", "moving_object.txt")) as f:
        moving_label = os.path.splitext(f.read().strip())[0]

    matches = glob.glob(os.path.join(args.take_dir, f"objs{suffix}", f"*_{moving_label}.*"))
    if not matches:
        raise FileNotFoundError(
            f"No canonical object found for moving_label={moving_label!r} "
            f"in {os.path.join(args.take_dir, f'objs{suffix}')}"
        )

    canonical_path = matches[0]
    depth_npz = os.path.join(sample_dir, "aligned_depths.npz")
    mask_dir = os.path.join(sample_dir, f"tracking_template{suffix}", "moving_masks")
    true_depth = os.path.join(args.take_dir, "initial_depth.npy")

    W, H = Image.open(rgb_path).size

    if args.camera_calibration is None:
        f = 0.8 * max(H, W)
        K = np.array([
            [f, 0, W / 2],
            [0, f, H / 2],
            [0, 0, 1],
        ])
        source, camera_name = "f = 0.8 * max(H, W)", "-"
    else:
        with open(args.camera_calibration, "r") as fh:
            calibration = json.load(fh)

        if args.camera_name is not None:
            if args.camera_name not in calibration:
                raise KeyError(
                    f"Camera {args.camera_name!r} not found in {args.camera_calibration}. "
                    f"Available cameras: {list(calibration.keys())}"
                )
            camera_name = args.camera_name
        else:
            camera_name = next(iter(calibration))

        K = np.asarray(calibration[camera_name]["intrinsics"], dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"Invalid intrinsic matrix shape {K.shape} in {args.camera_calibration}")
        source = args.camera_calibration

    print("=" * 60)
    print("Camera intrinsics")
    print(f"  Source : {source}")
    print(f"  Camera : {camera_name}")
    print(f"  Image  : {W} x {H}")
    print("  K:")
    print(K)
    print("=" * 60)

    canonical, faces = load_canonical(canonical_path)

    mask0_path = os.path.join(mask_dir, "frame_00000.png")
    mask0 = np.array(Image.open(mask0_path)) > 0
    depths = prepare_depth(true_depth, mask0_path, depth_npz, W, H, dtype=np.dtype(args.depth_dtype).type)
    print(depths.shape)

    # ------------------------------------------------------------------
    # Canonical registration: G -> first cloud, then frame-to-frame ICP
    # ------------------------------------------------------------------
    pts0 = depth_to_pts(depths[0], mask0, K)

    canonical, scale = fit_canonical_to_bbox(canonical, pts0)
    print("BBOX aligned scale:", scale)

    T_init = pca_init(canonical, pts0)
    T, rmse, reg = icp_o3d(canonical, pts0, T_init, OUT_DIR=out_dir)
    canonical = transform_points(canonical, T)
    print("Init RMSE:", rmse)

    canonical_init = canonical.copy()

    all_depths = [pts0]
    all_vertices = [canonical]
    prev_T = np.eye(4)

    for t in range(1, len(depths)):
        mask_prev = np.array(Image.open(os.path.join(mask_dir, f"frame_{t-1:05d}.png"))) > 0
        mask_curr = np.array(Image.open(os.path.join(mask_dir, f"frame_{t:05d}.png"))) > 0

        pts_prev = depth_to_pts(depths[t - 1], mask_prev, K)
        pts_curr = depth_to_pts(depths[t], mask_curr, K)

        print(
            f"[Frame {t}] mask_prev={mask_prev.sum()}, mask_curr={mask_curr.sum()}, "
            f"pts_prev={len(pts_prev)}, pts_curr={len(pts_curr)}"
        )

        if len(pts_prev) < 20 or len(pts_curr) < 20:
            print(f"Frame {t}: insufficient points, freezing pose (reusing prev_T)")
            T = prev_T.copy()
        else:
            try:
                T, rmse, reg = icp_o3d(pts_prev, pts_curr, prev_T)
                print(f"Frame {t} RMSE:", rmse)
            except RuntimeError as e:
                print(f"Frame {t}: icp_o3d raised RuntimeError ({e}), freezing pose (reusing prev_T)")
                T = prev_T.copy()

        if not np.all(np.isfinite(T)):
            print(f"Frame {t}: non-finite transform T, reusing prev_T instead")
            T = prev_T.copy()

        prev_T = T.copy()
        all_depths.append(pts_curr if len(pts_curr) > 0 else all_depths[-1])
        all_vertices.append(transform_points(all_vertices[-1], T))

    # ------------------------------------------------------------------
    # Vertex <-> depth correspondences (one table per frame)
    # ------------------------------------------------------------------
    vertices_to_depth = []
    for depth_pts, canonical_pts in zip(all_depths, all_vertices):
        vertex_to_depth, _ = build_vertex_correspondence(canonical_pts, depth_pts)
        vertices_to_depth.append(vertex_to_depth)

    # Per frame: {vertex id: (registered vertex, mean of the depth points matched to it)}.
    # The same fixed set of vertex ids is then tracked over the whole rollout.
    total_vertices = defaultdict(dict)
    for frame in tqdm.tqdm(range(len(vertices_to_depth))):
        depth_pts = all_depths[frame]
        rigid_vertices = all_vertices[frame]
        frame_map = {}
        for vid, correspondences in vertices_to_depth[frame].items():
            if vid >= len(rigid_vertices):
                continue
            obs_sum = np.zeros(3, dtype=np.float64)
            count = 0
            for depth_id, _dist in correspondences:
                if depth_id < 0 or depth_id >= len(depth_pts):
                    continue
                obs_sum += depth_pts[depth_id]
                count += 1
            if count == 0:
                continue
            frame_map[vid] = (rigid_vertices[vid], obs_sum / count)
        total_vertices[frame] = frame_map

    num_frames = len(all_depths)

    # Seed the tracked vertex set from frame 0, subsampled evenly by vertex id if capped.
    base_vids = sorted(total_vertices[0].keys())
    if len(base_vids) == 0:
        raise RuntimeError("Frame 0 has no vertex-depth correspondences, cannot seed vertex tracking.")

    cap = len(base_vids) if not np.isfinite(max_gp_vertices) else min(len(base_vids), int(max_gp_vertices))
    if cap < len(base_vids):
        pick = np.linspace(0, len(base_vids) - 1, cap, dtype=int)
        tracked_vids = [base_vids[i] for i in pick]
    else:
        tracked_vids = base_vids

    n_tracked = len(tracked_vids)
    source = np.zeros((n_tracked, num_frames, 3))
    target = np.zeros((n_tracked, num_frames, 3))

    for t in range(num_frames):
        frame_map = total_vertices[t]
        for i, vid in enumerate(tracked_vids):
            if vid in frame_map:
                rigid_pt, obs_pt = frame_map[vid]
            elif t > 0:
                rigid_pt, obs_pt = source[i, t - 1, :], target[i, t - 1, :]  # carry forward
            else:
                rigid_pt = obs_pt = np.zeros(3)
            source[i, t, :] = rigid_pt
            target[i, t, :] = obs_pt

    original_vertices = np.transpose(np.asarray(all_vertices), (1, 0, 2))
    warped_vertices = original_vertices.copy()

    # ------------------------------------------------------------------
    # Transportation map per vertex: rigid prior (Kabsch) + RBF-GP residual
    # ------------------------------------------------------------------
    print("Fitting rigid priors per vertex...")
    src_affine_batch = np.zeros_like(source)
    for vid in tqdm.tqdm(range(source.shape[0])):
        affine = AffineWarper()
        affine.fit(source[vid], target[vid])
        src_affine_batch[vid] = affine.predict(source[vid])

    print("Fitting all vertex GPs in one batched op...")
    batch_gp = BatchGPWarper(n_iter=500)  # raise n_iter if the fits look under-converged
    gp_output = batch_gp.fit_predict(src_affine_batch, target)  # (n_tracked, num_frames, 3)

    tracked_ids_arr = np.asarray(tracked_vids, dtype=int)
    warped_vertices[tracked_ids_arr, :, :] = gp_output

    # Vertices with no depth match inherit the mean correction of the tracked ones.
    unmatched_ids = np.setdiff1d(np.arange(warped_vertices.shape[0]), tracked_ids_arr)
    for t in range(num_frames):
        V_prev = original_vertices[tracked_ids_arr, t]
        V_curr = warped_vertices[tracked_ids_arr, t]
        delta = V_curr.mean(axis=0) - V_prev.mean(axis=0)
        warped_vertices[unmatched_ids, t] += delta

    warped_vertices = np.transpose(warped_vertices, (1, 0, 2))  # (num_frames, N, 3)

    # ------------------------------------------------------------------
    # Rigid projection onto SE(3) . G (Kabsch) and end-effector trajectory
    # ------------------------------------------------------------------
    final_vertices = [canonical_init.copy()]
    current_vertices = canonical_init.copy()
    for t in range(1, len(warped_vertices)):
        current_vertices = current_vertices + (warped_vertices[t] - warped_vertices[t - 1])
        final_vertices.append(current_vertices.copy())
    final_vertices = np.asarray(final_vertices)

    V_ref = final_vertices[0]
    recovered_T = []
    for V_t in final_vertices:
        R, t_vec = kabsch(V_ref, V_t)
        T_rec = np.eye(4)
        T_rec[:3, :3] = R
        T_rec[:3, 3] = t_vec
        recovered_T.append(T_rec)

    centroid = final_vertices[0].mean(axis=0)
    T_ee_0 = np.eye(4)
    T_ee_0[:3, 3] = centroid

    T_grasp = np.linalg.inv(recovered_T[0]) @ T_ee_0
    trajectory_ee = [T @ T_grasp for T in recovered_T]
    print("EE trajectory (centroid-based) built")

    np.save(os.path.join(out_dir, f"trajectory_ee{output_suffix}.npy"), np.stack(trajectory_ee))
    print(f"Saved trajectory_ee{output_suffix}.npy")

    visualize_vertex_flow_on_image(
        image_path=rgb_path,
        canonical=canonical,
        K=K,
        recovered_T=recovered_T,
        frame_idx=None,
        stride=20,
        save_path=os.path.join(out_dir, f"actionable_flow{output_suffix}.png"),
    )
    visualize_vertex_trajectories_on_image(
        image_path=rgb_path,
        canonical=canonical,
        K=K,
        recovered_T=recovered_T,
        n_vertices=25,
        save_path=os.path.join(out_dir, f"vertex_flow_2d{output_suffix}.png"),
    )
    print("GenieMotion computation done")


if __name__ == "__main__":
    main()
