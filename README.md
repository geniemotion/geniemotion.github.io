# GenieMotion: code for the final pipeline stage

Anonymous code release accompanying the submission *GenieMotion: Canonical
Geometry-Grounded SE(3) Trajectory Recovery from Generated Videos*.

This repository contains the proposed trajectory-recovery stage (`genie_motion/main_sam2.py`)
and one sample whose upstream preprocessing is already done:

| Upstream step (done, not included) | Output shipped in `sample/` |
|---|---|
| Video generation | per-frame depth, aligned to the observation (`aligned_depths.npz`) |
| SAM3D canonical mesh from the initial RGB-D frame | `objs_sam2/*.npz` |
| SAM2 segmentation and moving-object selection | `tracking_template_sam2/` |

`main_sam2.py` then performs:

1. canonical registration, a single mesh registered to the first depth cloud and propagated by point-to-plane ICP (Alg. 1, lines 5-8);
2. per-vertex transportation maps, a rigid prior plus an RBF Gaussian process on the residual (lines 9-13);
3. projection onto the SE(3)-orbit of the canonical mesh by Kabsch alignment (lines 14-16);
4. composition with the grasp transform to give the end-effector trajectory.

## Setup

```
pip install -r requirements.txt
```

A CUDA GPU is recommended (the batched GP fit runs on GPU if one is available, otherwise on CPU).

## Run

```
cd genie_motion
python main_sam2.py \
    --take-dir ../sample \
    --sample sample_15
```

Outputs are written to `sample/sample_15/`:

- `trajectory_ee_sam2_allGP.npy`: (T, 4, 4) end-effector poses
- `actionable_flow_sam2_allGP.png`: per-vertex flow overlaid on the first frame
- `vertex_flow_2d_sam2_allGP.png`: per-vertex trajectories overlaid on the first frame
- `T_sim3.npy`: frame-0 canonical-to-observation registration

Depth is handled as `uint16` (millimetres, as from a RealSense); pass `--depth-dtype float32` for simulator depth.

Camera intrinsics: by default a pinhole model with `f = 0.8 * max(H, W)` and the principal point at the image centre is used. A calibration JSON can be supplied instead with `--camera-calibration` (`{camera: {"intrinsics": 3x3}}`) and optionally `--camera-name`.

## Layout

```
genie_motion/
  main_sam2.py              entry point
  alignment.py              PCA initialisation and point-to-plane ICP (Open3D)
  vertex_correspondence.py  depth point -> nearest mesh vertex
  affine_transform.py       per-vertex rigid prior (Kabsch / Procrustes)
  warping_transform_gpu.py  batched RBF-GP residual fit (GPyTorch)
  get_trajectory.py         Kabsch
  utils.py                  mesh / depth loading and geometry helpers
  visualize.py              2D overlays
sample/                     one fully preprocessed sample
```

## Notes

- Mesh format: `objs_sam2/` may hold a SAM3D `.pth` (requires the `sam3d_objects` package to load; it is decimated to 1/5 of its triangles on load), or an already-decimated plain mesh (`.npz` with `vertices` and `faces`, or `.ply` / `.obj`), which is used as-is. The sample ships the plain `.npz`, so no extra package is needed.
- Upstream stages (video generation, SAM3D, SAM2 tracking, depth estimation) use third-party models and are not part of this release.
- The paper's experiments additionally rely on internal registration (ICP) variants and supporting tools that are not part of this release. The full version will be made public once the decision is out.
