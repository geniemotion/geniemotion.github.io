import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: figures are saved, never shown
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageDraw as ImageDraw

from utils import transform_points


def _project(K, pts_3d):
    """(N, 3) camera-frame points -> (N, 2) integer pixel coordinates."""
    X, Y, Z = pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2]
    Z = np.clip(Z, 1e-6, None)
    u = (K[0, 0] * X / Z + K[0, 2]).astype(int)
    v = (K[1, 1] * Y / Z + K[1, 2]).astype(int)
    return np.stack([u, v], axis=-1)


def visualize_vertex_flow_on_image(
    image_path,
    canonical,
    K,
    recovered_T,
    frame_idx=None,
    stride=30,
    save_path=None,
):
    """
    Project canonical mesh vertices into the image plane and draw per-vertex
    flow lines (frame 0 -> frame_idx) coloured by magnitude.

    Args:
        image_path  : path to the start RGB image
        canonical   : (N, 3) canonical mesh vertices (aligned to frame 0)
        K           : (3, 3) camera intrinsics
        recovered_T : list of (4, 4) SE(3) transforms, one per frame
        frame_idx   : frame to visualise (default: last)
        stride      : vertex subsampling for clarity
        save_path   : if set, the figure is saved here
    """
    if frame_idx is None:
        frame_idx = len(recovered_T) - 1

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H_img, W_img = img.shape[:2]

    V0 = canonical[::stride]
    Vt = transform_points(V0, recovered_T[frame_idx])

    uv0 = _project(K, V0)
    uvt = _project(K, Vt)

    flow_mag = np.linalg.norm(uvt - uv0, axis=1)
    flow_mag_norm = (flow_mag - flow_mag.min()) / (flow_mag.max() - flow_mag.min() + 1e-8)

    cmap = matplotlib.colormaps["plasma"]

    overlay_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(overlay_pil)

    for i in range(len(uv0)):
        x0, y0 = uv0[i]
        xt, yt = uvt[i]

        if not (0 <= x0 < W_img and 0 <= y0 < H_img):
            continue
        if not (0 <= xt < W_img and 0 <= yt < H_img):
            continue

        rgba = cmap(flow_mag_norm[i])
        color = tuple(int(c * 255) for c in rgba[:3])

        draw.line([(x0, y0), (xt, yt)], fill=color, width=2)

        r = 3
        draw.ellipse([(xt - r, yt - r), (xt + r, yt + r)], fill=color)

    result = np.array(overlay_pil)

    plt.figure(figsize=(10, 6))
    plt.imshow(result)
    plt.axis("off")
    plt.title(f"Per-Vertex Actionable Flow - Frame {frame_idx}")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved flow visualization: {save_path}")

    plt.close()
    return result


def visualize_vertex_trajectories_on_image(
    image_path,
    canonical,
    K,
    recovered_T,
    n_vertices=20,
    save_path=None,
):
    """
    Draw the full per-vertex trajectory across all frames on the image.
    One colour per vertex; line width grows toward the end to show direction.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H_img, W_img = img.shape[:2]

    idx = np.linspace(0, len(canonical) - 1, n_vertices, dtype=int)
    V_selected = canonical[idx]

    # (T, n_vertices, 2) pixel position of each selected vertex at each frame
    all_frame_uvs = np.array(
        [_project(K, transform_points(V_selected, T)) for T in recovered_T]
    )

    cmap = matplotlib.colormaps["rainbow"]
    vertex_colors = [
        tuple(int(c * 255) for c in cmap(i / n_vertices)[:3])
        for i in range(n_vertices)
    ]

    overlay = Image.fromarray(img)
    draw = ImageDraw.Draw(overlay)

    n_frames = len(recovered_T)

    for v in range(n_vertices):
        color = vertex_colors[v]

        for t in range(1, n_frames):
            x0, y0 = all_frame_uvs[t - 1, v]
            x1, y1 = all_frame_uvs[t, v]

            if not (0 <= x0 < W_img and 0 <= y0 < H_img):
                continue
            if not (0 <= x1 < W_img and 0 <= y1 < H_img):
                continue

            thickness = max(1, int(2 * t / n_frames) + 1)
            draw.line([(x0, y0), (x1, y1)], fill=color, width=thickness)

        x0, y0 = all_frame_uvs[0, v]
        if 0 <= x0 < W_img and 0 <= y0 < H_img:
            r = 5
            draw.ellipse(
                [(x0 - r, y0 - r), (x0 + r, y0 + r)],
                fill=color, outline=(255, 255, 255),
            )

        xT, yT = all_frame_uvs[-1, v]
        if 0 <= xT < W_img and 0 <= yT < H_img:
            r = 4
            draw.ellipse(
                [(xT - r, yT - r), (xT + r, yT + r)],
                fill=(255, 255, 255), outline=color,
            )

    result = np.array(overlay)

    plt.figure(figsize=(12, 7))
    plt.imshow(result)
    plt.axis("off")
    plt.title("Per-Vertex 4D Flow Trajectories")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved: {save_path}")

    plt.close()
    return result
