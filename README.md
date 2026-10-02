# PointWorld-DROID gripper points

Goal: per-frame 2D points on the **gripper only**, for the **full trajectory** of **every
PointWorld-DROID episode**, in each exterior camera, with a per-point visibility flag.

This repository holds the prototype code as it exists today plus the instructions needed to
finish the pipeline. The prototype labels one episode at a time correctly. A batch driver
that runs it over the whole dataset and writes results to disk has **not** been written yet
(see [What is needed to complete the pipeline](#what-is-needed-to-complete-the-pipeline)).

## How it works

No video and no point tracker are used. Gripper points come from robot kinematics,
projection comes from PointWorld's calibrated cameras, and visibility comes from
PointWorld's per-frame depth.

1. **Gripper model.** About 116 points are sampled once on the surface of a Robotiq 2F-85
   URDF (area-weighted across links, with surface normals). Each point keeps a fixed identity,
   so the output is a set of point trajectories.
2. **Pose per step.** For every step of the raw DROID `trajectory.h5`, the gripper base is
   placed at `observation/robot_state/cartesian_position` (xyz + extrinsic `xyz` Euler) and
   the fingers are opened with `observation/robot_state/gripper_position`.
3. **Projection.** Points are projected into each exterior camera using PointWorld's
   optimized extrinsics (`cameras/<ep>_cameras.json`) and intrinsics (from the first clip in
   `flows-fs-optimized/<ep>_flows.h5`). The intrinsics are at **320x180**, so native output
   coordinates are 320x180 pixels.
4. **Visibility.** A point is visible when its surface normal faces the camera **and**
   PointWorld's per-frame 320x180 depth (`depth_320x180/<ep>_depth.h5`) shows nothing
   significantly in front of it (tolerance `0.03 m + 0.03 * z`). This covers occlusion by
   objects, the arm, and the gripper itself.
5. **Quality score.** `match_rate` is the fraction of front-facing, in-frame points whose
   projected depth agrees with observed depth. A low score flags a camera whose calibration
   is off.

### Why raw DROID is required

PointWorld flow files only store robot state inside short clip windows (for example
`5:16`, `10:21`, then a gap to `65:76`). The full per-step pose is only in the raw DROID
`trajectory.h5`. Raw MP4s are **not** needed for labels; they are only used by the
visual check script.

## Files

| File | Role | Needed for the full run |
|---|---|---|
| `fk_gripper_labels.py` | `GripperModel` (point sampling, finger kinematics) and `label_episode(ep, model, ...)` (projection, visibility, quality score for every step and exterior camera). Its `__main__` only runs a calibration sweep over mount yaw and prints scores. | **Yes** |
| `render_fk.py` | Labels one episode, saves an `.npz`, and renders both exterior cameras side by side on the raw MP4s (green visible, red hidden, gray no depth). Needs OpenCV, ffmpeg and the raw MP4s. | No (checking only) |
| `sample_eval.py` | Coverage scan of all PointWorld episodes against local raw data, then labels 40 random episodes and prints quality and runtime statistics. | No (evaluation only) |

## Requirements

### Python

Python 3.11 with:

```
numpy h5py scipy yourdfpy trimesh
```

`render_fk.py` additionally needs `opencv-python` and an ffmpeg binary. Its ffmpeg path is
hard-coded (`FF = ...imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2`); point it at any
ffmpeg. No GPU is needed.

On the cluster this was developed on, the `cuteanything` conda env already has everything.

### Gripper model (URDF + meshes)

The code uses the Robotiq 2F-85 description from `a-price/robotiq_arg85_description`
(URDF plus 10 STL meshes). Fetch it either way:

```bash
git clone https://github.com/a-price/robotiq_arg85_description
# or
pip install robot_descriptions
python -c "from robot_descriptions import robotiq_2f85_description"   # clones into ~/.cache/robot_descriptions/
```

Then set `URDF_DIR` in `fk_gripper_labels.py` to the cloned folder. It currently points at
`~/.cache/robot_descriptions/robotiq_arg85_description`, which is a cache location; copying
it next to the code is safer.

### Data

Paths are hard-coded at the top of `fk_gripper_labels.py`:

| Constant | Points to | Files read |
|---|---|---|
| `PW` | restored PointWorld-DROID `droid/` root | `cameras/<ep>_cameras.json`, `depth_320x180/<ep>_depth.h5`, `flows-fs-optimized/<ep>_flows.h5` (intrinsics only) |
| `RAW` | raw DROID `droid_raw/1.0.1/` | `<scene_path>/trajectory.h5` (located via `scene_path` in the camera JSON) |

Coverage measured on the development cluster: **42,927 of 42,935** PointWorld episodes have a
local raw trajectory. The other 8 will raise `FileNotFoundError`.

## Conventions (verified, not to be changed)

| Item | Value | How it was established |
|---|---|---|
| PointWorld `optimized_extrinsics` | **world-to-camera** | Depth agreement, both directions tested |
| Gripper mount yaw about base z | **90 degrees** | Best depth agreement over 0, ±45, 90 degrees |
| Intrinsics resolution | 320x180 | Matches depth maps |
| Frame index | raw trajectory step = depth frame = raw MP4 frame | Zero time shift fits best against an independent projection |
| PointWorld / earlier tracker pipeline | use every 2nd step (`canonical_timestamps`) | Root attrs of flow files |
| Finger joint | `finger_joint = clip(gripper_position, 0, 1) * 0.725` | Approximation (URDF limit 0.725) |

**Important:** `label_episode` defaults are wrong for this data. Always call

```python
label_episode(ep, model, mount_yaw_deg=90, ext_is_cam_to_world=False)
```

The default `mount_yaw_deg=0` misplaces the fingers, and `ext_is_cam_to_world=None` tries
both extrinsic directions, which doubles the work.

## Validation so far

| Check | Result |
|---|---|
| Distance to an independent kinematic projection (earlier pipeline's `fk_uv`), one episode | median 2.6 px, max 4.4 px at 854x480 |
| Runtime | about 0.65 s per episode on one CPU core |
| `match_rate` over 200 cameras (100 random episodes) | p5 0.41, p10 0.48, median 0.69 |
| Cameras with `match_rate` below 0.4 | 5% |
| Cameras with `match_rate` below 0.5 | 13.5% |

Low scores have two causes, seen in renders:

- **Bad calibration.** On `PennPAL+06b0ffa5+2023-04-29-23h-43m-49s`, camera `25455306`, the
  points sit visibly beside the real gripper and the score is 0.00. Drop or re-calibrate.
- **Overly strict visibility.** On `TRI+52ca9b6a+2024-02-12-15h-17m-09s`, camera `28813166`,
  points sit on the gripper but it is seen edge-on, and the depth test marks most as hidden.
  Labels are right; visibility is conservative.

## What is needed to complete the pipeline

### 1. Output resolution (one-line change)

`OUT_HW = (480, 854)` rescales coordinates to 854x480 to match the earlier tracker pipeline.
For 320x180 output set `OUT_HW = (180, 320)`, or remove the rescale on the line that builds
`uv=` in `label_episode`.

### 2. Batch driver (new script, not written yet)

Write `run_all.py` that:

1. Lists episodes from `PW/cameras/*_cameras.json` (the PointWorld episode list).
2. Builds one `GripperModel()` per worker process (loading the URDF and sampling meshes has a
   startup cost, so do not rebuild it per episode).
3. For each episode, skips it if the output file exists, then calls
   `label_episode(ep, model, mount_yaw_deg=90, ext_is_cam_to_world=False)`.
4. Saves one `np.savez_compressed` per episode, for example `out/<ep>_gripper.npz`, with for each
   exterior camera serial `s`:

   | Key | Shape | Meaning |
   |---|---|---|
   | `{s}_uv` | (T, N, 2) float32 | pixel coordinates (320x180 after step 1) |
   | `{s}_z` | (T, N) float16 | depth of the point in the camera frame, metres |
   | `{s}_in_frame` | (T, N) bool | projects inside the image with positive depth |
   | `{s}_visible` | (T, N) bool | front-facing and not occluded per depth |
   | `{s}_known` | (T, N) bool | depth was available at that pixel |
   | `{s}_match_rate` | () float | quality score |

   plus once per file: `serials`, `link` (N,) link name per point, `points_in_link` (N, 3).
   `render_fk.py` already shows the `savez_compressed` pattern for most of these keys.
5. Catches exceptions per episode (the 8 missing raw episodes, any corrupt file) and appends
   them to an error log instead of stopping.
6. Writes a summary table (episode, serial, T, match_rate, fraction of frames with any visible
   point) for QC.

### 3. Parallelism

About 8 CPU-hours total (42,927 episodes at 0.65 s). Use a Slurm array where each task takes
a contiguous slice of the sorted episode list, or a `multiprocessing.Pool` with one
`GripperModel` per worker (`initializer=`). Memory is small: one camera's depth stack is about
120 MB.

### 4. Quality control

Use `match_rate` to filter cameras. Spot-check renders (`render_fk.py <ep> <out_dir> 4`)
for cameras around 0.4 to 0.5 before fixing a threshold. For rejected cameras, either drop
them or re-fit their extrinsics by aligning the projected gripper with depth.

## Known limitations

- The 90 degree mount yaw was chosen on one episode and held up in the samples viewed. Verify
  it per lab if a lab uses a different mount.
- The finger opening mapping is linear and approximate.
- Visibility from 320x180 depth is conservative for thin finger parts and edge-on views.
- Only exterior cameras are labeled. PointWorld has no wrist-camera flows.
