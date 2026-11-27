from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from huggingface_hub import HfApi, RepoFile, RepoFolder

from scripts import download


REVISION = "a" * 40
SCENE = "1K/scene-a"


def folder(path):
    return RepoFolder(path=path, oid=REVISION)


def file(path, size=5):
    return RepoFile(path=path, oid=REVISION, size=size)


class DownloadSelectionTests(unittest.TestCase):
    def setUp(self):
        self.api = Mock(spec=HfApi)

    def test_defaults_to_published_groups_and_rejects_unavailable_groups(self):
        def list_groups(*args, **kwargs):
            return iter([folder("1K"), folder("3K"), folder("HABI-Bench"), folder("other")])

        self.api.list_repo_tree.side_effect = list_groups
        self.assertEqual(
            download.published_groups(self.api, REVISION, None), ["1K", "3K", "HABI-Bench"]
        )
        with self.assertRaisesRegex(ValueError, "Groups unavailable: 2K"):
            download.published_groups(self.api, REVISION, ["1K", "2K"])

    def test_rgb_only_looks_up_videos_without_recursing_into_depth(self):
        rgb_path = f"{SCENE}/rgb.mkv"
        self.api.list_repo_tree.return_value = iter([folder(SCENE), file("1K/README.md")])
        self.api.get_paths_info.return_value = [file(rgb_path, 1024)]

        selection = download.iter_files(self.api, REVISION, ["1K"], ["rgb"])

        self.assertEqual(download.count_files(selection), (1, 1024))
        self.api.list_repo_tree.assert_called_once_with(
            download.REPO_ID, "1K", repo_type="dataset", revision=REVISION, recursive=False
        )
        self.api.get_paths_info.assert_called_once_with(
            download.REPO_ID, [rgb_path], repo_type="dataset", revision=REVISION
        )

    def test_depth_and_metadata_exclude_rgb_and_normal(self):
        depth_path = f"{SCENE}/depth/000000.exr"
        metadata_path = f"{SCENE}/transforms.json"
        self.api.list_repo_tree.return_value = iter([
            folder(f"{SCENE}/depth"),
            file(depth_path),
            file(metadata_path),
            file(f"{SCENE}/rgb.mkv"),
            file(f"{SCENE}/normal.mkv"),
            file(f"{SCENE}/depth/README.md"),
        ])

        selection = download.iter_files(self.api, REVISION, ["1K"], ["depth", "metadata"])

        self.assertEqual([entry.path for entry in selection], [depth_path, metadata_path])
        self.api.get_paths_info.assert_not_called()

    @patch("huggingface_hub.hf_hub_download")
    def test_only_selected_files_are_downloaded(self, hub_download):
        rgb = file(f"{SCENE}/rgb.mkv")
        output = Path("HABI-12K")
        self.assertEqual(list(download.download_files([rgb], REVISION, output, 1)), [rgb])
        hub_download.assert_called_once_with(
            download.REPO_ID, rgb.path, repo_type="dataset", revision=REVISION,
            local_dir=output, endpoint=download.ENDPOINT, token=False,
        )
