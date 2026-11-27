"""Read synchronized panoramas and camera poses from a scene directory."""

from collections.abc import Iterable, Iterator
from contextlib import ExitStack, closing
import json
from pathlib import Path

import av
import numpy as np
import OpenEXR


MODALITIES = ("rgb", "depth", "normal")
VIDEO_MODALITIES = ("rgb", "normal")
Frame = dict[str, float | int | np.ndarray]


class Scene:
    """A directory containing one packaged HABI-12K trajectory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with (self.path / "transforms.json").open(encoding="utf-8") as handle:
            self.metadata = json.load(handle)

        self._frames = self.metadata["frames"]
        if not self._frames or len(self._frames) != self.metadata["frame_count"]:
            raise ValueError(f"{self.path}: inconsistent frame count in transforms.json")
        if any(frame["index"] != index for index, frame in enumerate(self._frames)):
            raise ValueError(f"{self.path}: expected contiguous frame indices starting at zero")

        settings = self.metadata["settings"]
        self._shape = (settings["height"], settings["width"])
        self._far_clip_m = (
            self.metadata["clip_end_scene_units"] * self.metadata["meters_per_blender_unit"]
        )

    def __len__(self) -> int:
        return len(self._frames)

    def read_frame(self, index: int, *, modalities: Iterable[str] = MODALITIES) -> Frame:
        """Read one frame by its zero-based index."""
        with closing(self.iter_frames(start=index, stop=index + 1, modalities=modalities)) as frames:
            return next(frames)

    def iter_frames(
        self, start: int = 0, stop: int | None = None, *, modalities: Iterable[str] = MODALITIES
    ) -> Iterator[Frame]:
        """Stream selected modalities in display order over the half-open range [start, stop)."""
        stop = len(self) if stop is None else stop
        if not 0 <= start <= stop <= len(self):
            raise IndexError(f"{self.path}: frame range [{start}, {stop}) outside [0, {len(self)})")

        requested = set(modalities)
        unknown = requested.difference(MODALITIES)
        if unknown:
            raise ValueError(f"Unknown modalities: {', '.join(sorted(unknown))}")
        if start == stop:
            return

        with ExitStack() as stack:
            videos = {
                name: self._open_video(stack, name) for name in VIDEO_MODALITIES if name in requested
            }
            for index in range(stop):
                images = {}
                for name, video in videos.items():
                    image = next(video, None)
                    if image is None:
                        path = self.path / f"{name}.mkv"
                        raise ValueError(f"{path}: missing frame {index}")
                    if index >= start:
                        images[name] = self._video_rgb(image, name, index)
                if index >= start:
                    yield self._assemble_frame(index, images, requested)

            if stop == len(self):
                for name, video in videos.items():
                    if next(video, None) is not None:
                        path = self.path / f"{name}.mkv"
                        raise ValueError(f"{path}: extra frames after {stop}")

    def validate(self) -> None:
        """Check the depth filenames and decode every frame in the scene."""
        expected = {f"{index:06d}.exr" for index in range(len(self))}
        actual = {path.name for path in (self.path / "depth").glob("*.exr")}
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"{self.path / 'depth'}: missing {missing[:5]}, extra {extra[:5]}")
        for _ in self.iter_frames():
            pass

    def _open_video(self, stack: ExitStack, name: str) -> Iterator[av.VideoFrame]:
        path = self.path / f"{name}.mkv"
        container = stack.enter_context(av.open(str(path)))
        streams = container.streams.video
        if len(streams) != 1 or streams[0].codec_context.name != "hevc":
            raise ValueError(f"{path}: expected one HEVC video stream")
        return container.decode(video=0)

    def _video_rgb(self, frame: av.VideoFrame, name: str, index: int) -> np.ndarray:
        if (frame.height, frame.width) != self._shape or frame.format.name != "yuv444p10le":
            path = self.path / f"{name}.mkv"
            raise ValueError(f"{path}: unexpected dimensions or pixel format at frame {index}")
        return frame.to_ndarray(format="rgb24")

    def _read_depth(self, index: int) -> np.ndarray:
        path = self.path / "depth" / f"{index:06d}.exr"
        with OpenEXR.File(str(path), separate_channels=True) as image:
            channel = image.channels().get("V")
            if channel is None:
                raise ValueError(f"{path}: missing V depth channel")
            pixels = channel.pixels
            if pixels.shape != self._shape or pixels.dtype != np.float16:
                raise ValueError(f"{path}: unexpected depth dimensions or dtype")
            return pixels.astype(np.float32)

    def _assemble_frame(
        self, index: int, images: dict[str, np.ndarray], requested: set[str]
    ) -> Frame:
        metadata = self._frames[index]
        pose = np.asarray(metadata["camera_to_world_cv_m"], dtype=np.float32)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"{self.path}: invalid camera_to_world_cv_m at frame {index}")
        result: Frame = {
            "index": index,
            "timestamp_seconds": float(metadata["timestamp_seconds"]),
            "camera_to_world_cv_m": pose,
        }
        if "rgb" in requested:
            result["rgb"] = images["rgb"]
        if requested.intersection(("depth", "normal")):
            depth = self._read_depth(index)
            valid = np.isfinite(depth) & (depth > 0) & (depth < self._far_clip_m)
            result["valid_mask"] = valid
            if "depth" in requested:
                result["depth_m"] = depth
            if "normal" in requested:
                normal = images["normal"].astype(np.float32) / 127.5 - 1.0
                normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
                normal[~valid] = 0
                result["normal_world"] = normal
        return result
