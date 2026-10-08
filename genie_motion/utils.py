import os
import sys
import types

import cv2
import numpy as np
import open3d as o3d
import torch

# SAM3D .pth files pickle references into the sam3d_objects package; register an
# empty stand-in module so they can be unpickled when only the mesh is needed.
init_module = types.ModuleType("sam3d_objects.init")
sys.modules["sam3d_objects.init"] = init_module


# ==============================
# Canonical mesh
# ==============================
def load_canonical(path):
    """
    Load the canonical mesh as (vertices, faces).

    A SAM3D output (.pth, needs the sam3d_objects package to unpickle) is
    decimated to 1/5 of its triangles. A plain mesh (.npz with 'vertices' and
    'faces', or .ply/.obj/.stl) is used as-is and is expected to be decimated
    already.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pth":
        pts, triangles = decimate_mesh(*_read_sam3d_mesh(path))
    elif ext == ".npz":
        data = np.load(path)
        pts, triangles = data["vertices"], data["faces"]
    else:
        raw = o3d.io.read_triangle_mesh(path)
        pts, triangles = np.asarray(raw.vertices), np.asarray(raw.triangles)

    print(pts.shape, triangles.shape)
    return pts, triangles


def _read_sam3d_mesh(path):
    out = torch.load(path, map_location="cpu", weights_only=False)
    mesh = out["mesh"][0] if isinstance(out["mesh"], list) else out["mesh"]
    return mesh.vertices.detach().cpu().numpy(), mesh.faces.detach().cpu().numpy()


def decimate_mesh(pts, triangles):
    """Quadric decimation to 1/5 of the triangles."""
    size = int(len(triangles)//5)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=size)

    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


def fit_canonical_to_bbox(canonical, depth_pts, do_scale=True):
    """
    Align canonical mesh to depth bounding box (scale + translate only)
    """
    c_center = canonical.mean(axis=0)
    d_center = depth_pts.mean(axis=0)
    c_min, c_max = canonical.min(axis=0), canonical.max(axis=0)
    d_min, d_max = depth_pts.min(axis=0), depth_pts.max(axis=0)

    c_size = c_max - c_min
    d_size = d_max - d_min
    scale = np.max(d_size) / (np.max(c_size) + 1e-8)
    if do_scale:
        canonical_aligned = (canonical - c_center) * scale + d_center
    else:
        canonical_aligned = (canonical - c_center) + d_center

    return canonical_aligned, scale


def transform_points(pts, T):
    return (T[:3,:3] @ pts.T).T + T[:3,3]


# ==============================
# Depth
# ==============================
def process_video_depth(
    pred_npy, alpha, beta, width: int = 1280, height: int = 720, dtype=np.float32
):
    """
    Resize every predicted depth frame to (width, height), apply the metric
    scale/shift (alpha * d + beta), clip negatives to zero and cast to `dtype`.

    dtype: np.uint16 for real RealSense depth, which is stored as integer
    millimetres; np.float32 for simulator depth, which is stored as floating
    point values.
    """
    processed_depth = []

    for frame in pred_npy:
        d = cv2.resize(
            frame.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC
        )

        d = alpha * d + beta
        d[d < 0] = 0
        processed_depth.append(d.astype(dtype))
    return np.array(processed_depth)


def find_center_of_mask(mask_path: str, window_size: int = 20) -> np.ndarray:
    """
    Returns an array of (row, col) pixel coordinates within a square window
    of side `window_size` centered on the mask's centroid.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(mask > 0)
    center_r, center_c = int(np.median(ys)), int(np.median(xs))

    half = window_size // 2
    h, w = mask.shape
    coords = []
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            r = center_r + dr
            c = center_c + dc
            if 0 <= r < h and 0 <= c < w:
                coords.append([r, c])
    return np.array(coords)


def find_scale_and_shift(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    pixel_coords: np.ndarray,
    mask_invalid: bool = False,
) -> tuple[float, float]:
    """
    Estimate alpha, beta s.t. alpha * depth_pred + beta ≈ depth_gt
    using only the pixels specified by pixel_coords (row,col).
    """
    # resize first frame of prediction to match ground-truth resolution
    pred0 = cv2.resize(
        depth_pred[0].astype(np.float32),
        (depth_gt.shape[1], depth_gt.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    rows = pixel_coords[:, 0]
    cols = pixel_coords[:, 1]

    dp_vals = pred0[rows, cols]
    dg_vals = depth_gt[rows, cols].astype(np.float32)

    if mask_invalid:
        valid = (dp_vals > 0) & (dg_vals > 0)
        dp_vals = dp_vals[valid]
        dg_vals = dg_vals[valid]

    # solve [dp_vals, 1] * [alpha; beta] = dg_vals
    A = np.stack([dp_vals, np.ones_like(dp_vals)], axis=-1)
    coefs, *_ = np.linalg.lstsq(A, dg_vals, rcond=None)
    alpha, beta = float(coefs[0]), float(coefs[1])
    return alpha, beta


def prepare_depth(
    gt_depth_path,
    mask_path,
    predicted_depth_path,
    width,
    height,
    dtype
):
    """
    Align the predicted depth video to metric depth.

    The scale and shift (alpha, beta) are fitted by least squares between the
    first predicted frame and the observed depth `gt_depth_path`, over a
    20 x 20 pixel window centred on the moving-object mask (`mask_path`).
    All frames of `predicted_depth_path` (npz with a 'depths' array) are then
    resized to (width, height) and converted with alpha * d + beta.

    dtype: np.uint16 for real RealSense depth (integer millimetres) or
    np.float32 for simulator depth.

    Returns:
        (T, height, width) array of metric depth.
    """
    gt = np.load(gt_depth_path)
    depth_npy = np.load(predicted_depth_path)["depths"]
    pixel_coords = find_center_of_mask(mask_path, window_size=20)
    alpha, beta = find_scale_and_shift(depth_npy, gt, pixel_coords, mask_invalid=True)
    return process_video_depth(depth_npy, alpha, beta, width, height, dtype=dtype)


def depth_to_pts(depth, mask, K):
    ys, xs = np.where(mask)
    z = depth[ys, xs]

    X = (xs - K[0,2]) / K[0,0] * z
    Y = (ys - K[1,2]) / K[1,1] * z

    return np.stack([X, Y, z], axis=-1)
