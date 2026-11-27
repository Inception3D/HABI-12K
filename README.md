# HABI-12K

HABI-12K is a synthetic indoor 3D dataset with 12,081 scenes and 2.16 million `2048 × 1024` panoramas. Each observation pairs RGB with metric depth, world-space shading normals, and a camera pose along a continuous trajectory. HABI-Bench contains 117 held-out scenes for camera estimation, depth prediction, and novel view synthesis.

**Paper** · Coming soon | **Dataset** · [Hugging Face](https://huggingface.co/datasets/Inception3D/HABI-12K) | **Project page** · [HABI-12K](https://sibo2rr.github.io/HABI-12K/)

## Download

Run the downloader from the cloned repository with Python 3.10 or later.

```bash
python -m pip install 'huggingface_hub>=1.0,<2'

# Download all files in the published 1K group.
python scripts/download.py --groups 1K

# Download only RGB videos from all published groups.
python scripts/download.py --modalities rgb

# Download RGB videos and camera metadata from 1K for use with the loader.
python scripts/download.py --groups 1K --modalities rgb metadata

# Preview the number and size of files for a selection.
python scripts/download.py --groups 1K --modalities rgb --dry-run
```

Use `--groups` to select one or more groups and `--modalities` to select any of `rgb`, `normal`, `depth`, and `metadata`. The loader requires `metadata` (`transforms.json`) and uses depth when reading normals. Omit `--groups` to include all currently published groups. Files keep their repository paths under `HABI-12K/`. Use `--output-dir` to choose another destination. Rerun the same command to resume a download via `https://hf-mirror.com`.

## Data format

The main dataset is organized into `1K` through `12K`, and the held-out scenes are under `HABI-Bench`. Every group uses the scene layout shown under `1K/`.

```text
HABI-12K/
├── 1K/
│   └── <scene_id>/
│       ├── rgb.mkv
│       ├── normal.mkv
│       ├── depth/
│       │   ├── 000000.exr
│       │   └── 000001.exr
│       └── transforms.json
├── 2K/
├── 3K/
├── 4K/
├── 5K/
├── 6K/
├── 7K/
├── 8K/
├── 9K/
├── 10K/
├── 11K/
├── 12K/
└── HABI-Bench/
    └── <scene_id>/
```

| File | Contents |
| --- | --- |
| `rgb.mkv` | RGB panoramas in a Matroska container with HEVC, `libx265`, CRF 6, and `yuv444p10le`. |
| `normal.mkv` | World-space shading normals encoded as RGB and stored with the same lossy video settings. After converting the video frame to RGB8, decode each vector as `2 * rgb / 255 - 1` and normalize it on valid pixels. |
| `depth/NNNNNN.exr` | Original OpenEXR depth for frame `NNNNNN`. The `V` channel stores float16 ray distance in meters. Depth is valid where it is finite, positive, and below the camera's far clip distance in meters. |
| `transforms.json` | Frame indices, trajectory timestamps, projection metadata, and camera transforms. `camera_to_world_cv_m` maps OpenCV camera coordinates (+X right, +Y down, +Z forward) to world coordinates in meters. Equirectangular frames have `intrinsic_matrix: null`. |

Video frames in display order and depth filenames use the same zero-based index as `frames[].index` in `transforms.json`. The videos are packaged at 30 fps, while the trajectories were sampled at 2 Hz. Use `frames[].timestamp_seconds` for the trajectory time. The `file_path` and `normal_path` fields retain the original PNG filenames, and the loader reads the corresponding frames from the two videos.

## Usage

Install the loader with Python 3.10 or later from the cloned repository.

```bash
python -m pip install .
```

Open a scene directory and stream aligned frames.

```python
from habi12k import Scene

scene = Scene("/path/to/HABI-12K/1K/<scene_id>")
for frame in scene.iter_frames(start=0, stop=min(8, len(scene))):
    rgb = frame["rgb"]                       # H × W × 3, uint8
    depth_m = frame["depth_m"]               # H × W, float32
    normal_world = frame["normal_world"]     # H × W × 3, float32
    valid = frame["valid_mask"]              # H × W, bool
    camera_to_world = frame["camera_to_world_cv_m"]  # 4 × 4, float32
    timestamp = frame["timestamp_seconds"]   # Seconds along the trajectory
```

`start` is inclusive and `stop` is exclusive. Omit `stop` to read to the end. For RGB-only reading, pass `modalities=("rgb",)`. Use `scene.read_frame(index)` for a single frame. The video decoders start at the beginning of the scene, so use one `iter_frames` call for a range of nearby frames. When requesting normals, the loader reads depth to build the validity mask and returns zero vectors at invalid pixels.

Validate a scene by checking its depth filenames and decoding both video streams to verify the expected frame counts and dimensions.

```bash
python -m habi12k validate "/path/to/HABI-12K/1K/<scene_id>"
```

## Citation
