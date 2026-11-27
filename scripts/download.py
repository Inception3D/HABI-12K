"""Download selected HABI-12K files from Hugging Face."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import islice
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huggingface_hub import HfApi, RepoFile


REPO_ID = "Inception3D/HABI-12K"
ENDPOINT = "https://hf-mirror.com"
GROUPS = (*(f"{index}K" for index in range(1, 13)), "HABI-Bench")
GROUP_NAMES = {group.lower(): group for group in GROUPS}
SCENE_FILES = {"rgb": "rgb.mkv", "normal": "normal.mkv", "metadata": "transforms.json"}
MODALITIES = (*SCENE_FILES, "depth")


def batches(items, size):
    remaining = iter(items)
    while batch := list(islice(remaining, size)):
        yield batch


def published_groups(api: HfApi, revision: str, groups: list[str] | None) -> list[str]:
    from huggingface_hub import RepoFolder

    entries = api.list_repo_tree(REPO_ID, repo_type="dataset", revision=revision)
    available_paths = {entry.path for entry in entries if isinstance(entry, RepoFolder)}
    available = [group for group in GROUPS if group in available_paths]
    if groups is None:
        return available
    missing = [group for group in groups if group not in available]
    if missing:
        raise ValueError(
            f"Groups unavailable: {', '.join(missing)}. "
            f"Published groups: {', '.join(available) or 'none'}."
        )
    return list(dict.fromkeys(groups))


def iter_files(
    api: HfApi, revision: str, groups: list[str], modalities: list[str]
) -> Iterator[RepoFile]:
    from huggingface_hub import RepoFile, RepoFolder

    filenames = {name for mode, name in SCENE_FILES.items() if mode in modalities}
    include_depth = "depth" in modalities
    for group in groups:
        entries = api.list_repo_tree(
            REPO_ID, group, repo_type="dataset", revision=revision, recursive=include_depth
        )
        if include_depth:
            for entry in entries:
                if isinstance(entry, RepoFile):
                    parts = entry.path.split("/")
                    scene_file = len(parts) == 3 and parts[-1] in filenames
                    depth_frame = (
                        len(parts) == 4 and parts[2] == "depth" and parts[-1].endswith(".exr")
                    )
                    if scene_file or depth_frame:
                        yield entry
        else:
            # Look up only the selected files without listing each scene's depth frames.
            paths = (
                f"{entry.path}/{name}"
                for entry in entries if isinstance(entry, RepoFolder)
                for name in sorted(filenames)
            )
            for batch in batches(paths, 100):
                matches = api.get_paths_info(REPO_ID, batch, repo_type="dataset", revision=revision)
                yield from (entry for entry in matches if isinstance(entry, RepoFile))


def count_files(files: Iterable[RepoFile]) -> tuple[int, int]:
    count = total_bytes = 0
    for file in files:
        count += 1
        total_bytes += file.size
    return count, total_bytes


def download_files(
    files: Iterable[RepoFile], revision: str, output_dir: Path, workers: int
) -> Iterator[RepoFile]:
    from huggingface_hub import hf_hub_download

    download = partial(
        hf_hub_download, REPO_ID, repo_type="dataset", revision=revision,
        local_dir=output_dir, endpoint=ENDPOINT, token=False,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for batch in batches(files, 100):
            for file, _ in zip(batch, executor.map(download, (file.path for file in batch))):
                yield file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups", nargs="+", choices=GROUPS,
        type=lambda name: GROUP_NAMES.get(name.lower(), name),
        help="Dataset groups to download (default: all published groups).",
    )
    parser.add_argument(
        "--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES),
        help="Files to download (default: all modalities, including camera metadata).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("HABI-12K"))
    parser.add_argument("--workers", type=int, default=4, help="Concurrent downloads (default: 4).")
    parser.add_argument(
        "--dry-run", action="store_true", help="Count selected files without downloading them."
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    # Set the mirror and disable proxy and native Xet settings for this process.
    for name in tuple(os.environ):
        if name.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}:
            del os.environ[name]
    os.environ["HF_ENDPOINT"] = ENDPOINT
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        import httpx
        from huggingface_hub import HfApi, set_client_factory
    except ImportError:
        parser.exit(1, "Install the downloader dependency with: pip install 'huggingface_hub>=1.0,<2'\n")

    set_client_factory(
        lambda: httpx.Client(trust_env=False, follow_redirects=True, timeout=60)
    )
    api = HfApi(endpoint=ENDPOINT, token=False)
    try:
        revision = api.repo_info(REPO_ID, repo_type="dataset", expand=["sha"]).sha
        if revision is None:
            raise ValueError("The dataset has no published revision.")
        groups = published_groups(api, revision, args.groups)
        if groups:
            print(
                f"{', '.join(groups)} | {', '.join(args.modalities)} | revision {revision[:12]}",
                flush=True,
            )
        files = iter_files(api, revision, groups, args.modalities)
        if not args.dry_run:
            files = download_files(files, revision, args.output_dir, args.workers)
        count, total_bytes = count_files(files)
        if count == 0:
            raise ValueError("No published files match the requested groups and modalities.")
    except (httpx.HTTPError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Download failed via {ENDPOINT}: {error}\n")

    action = "Matched" if args.dry_run else "Downloaded or already present"
    print(f"{action}: {count:,} files ({total_bytes / 1024**3:,.2f} GiB).")
    if not args.dry_run:
        print(f"Data directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
