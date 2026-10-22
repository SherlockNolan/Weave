"""Recorded dataset replay must stay lazy and preserve held-out provenance."""

import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/reproduce"))

from viser_dataset_catalog import MAX_STREAM_CHUNK, DatasetCatalog, read_frame_range


class BoundedReader:
    def __init__(self, stream):
        self.stream = stream
        self.max_request = 0

    def read(self, count=-1):
        if not 0 <= count <= MAX_STREAM_CHUNK:
            raise AssertionError(f"Unbounded compressed read: {count}")
        self.max_request = max(self.max_request, count)
        return self.stream.read(count)

    def seek(self, *_args):
        raise AssertionError("Compressed skipping must use bounded reads, not seek")


def visual_arrays(frames, joints=53, bodies=54):
    quats = np.zeros((frames, bodies, 4), dtype=np.float32)
    quats[:, :, 0] = 1
    return {
        "joint_pos": np.arange(frames * joints, dtype=np.float32).reshape(frames, joints),
        "body_pos_w": np.arange(frames * bodies * 3, dtype=np.float32).reshape(frames, bodies, 3),
        "body_quat_w": quats,
        "object_pos_w": np.arange(frames * 3, dtype=np.float32).reshape(frames, 3),
        "object_quat_w": quats[:, 0],
    }


class StreamingTests(unittest.TestCase):
    def test_late_compressed_frame_range_never_requests_the_whole_skipped_member(self):
        values = np.arange(80_000 * 53, dtype=np.float32).reshape(80_000, 53)
        npy = io.BytesIO()
        np.save(npy, values)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("joint_pos.npy", npy.getvalue())
        with zipfile.ZipFile(buffer) as archive, archive.open("joint_pos.npy") as stream:
            guarded = BoundedReader(stream)
            actual = read_frame_range(guarded, 79_950, 80_000)
            self.assertEqual(guarded.max_request, MAX_STREAM_CHUNK)
        np.testing.assert_array_equal(actual, values[79_950:])

    def test_fortran_storage_is_rejected_instead_of_selecting_wrong_frames(self):
        npy = io.BytesIO()
        np.save(npy, np.asfortranarray(np.arange(30).reshape(10, 3)))
        npy.seek(0)
        with self.assertRaisesRegex(ValueError, "C-contiguous"):
            read_frame_range(npy, 2, 4)


class CatalogTests(unittest.TestCase):
    def test_catalog_reads_metadata_only_and_selected_load_retains_final_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            path = repo / "data/test/tripod.npz"
            path.parent.mkdir(parents=True)
            expected = visual_arrays(8)
            np.savez_compressed(
                path,
                **expected,
                motion_names=["sub1_tripod_075_v00", "sub1_tripod_075_v01"],
                motion_lengths=[3, 5],
                object_names=["tripod", "tripod"],
                fps=[50],
            )
            # Catalog enumeration must not touch even one heavy frame member.
            getitem = np.lib.npyio.NpzFile.__getitem__

            def metadata_only(archive, key):
                if key not in {"motion_names", "motion_lengths", "object_names", "fps"}:
                    raise AssertionError(f"Eagerly loaded frame member: {key}")
                return getitem(archive, key)

            with patch.object(np.lib.npyio.NpzFile, "__getitem__", metadata_only):
                catalog = DatasetCatalog(repo)
            self.assertEqual(len(catalog.entries), 2)
            entry = catalog.search(query="v01 三脚架", split="test", hand="inspire")[0]
            self.assertEqual(entry.frame_start, 3)
            selected = catalog.load(entry.key)
            self.assertEqual(selected["source_split"].tolist(), ["test"])
            self.assertEqual(selected["motion_lengths"].tolist(), [5])
            for field in expected:
                np.testing.assert_array_equal(selected[field], expected[field][3:8])
            # A second load uses only selected cached buffers, not the archive.
            with patch("viser_dataset_catalog.read_frame_range", side_effect=AssertionError("Cache miss")):
                cached = catalog.load(entry)
            np.testing.assert_array_equal(cached["joint_pos"][-1], expected["joint_pos"][-1])
            # An interrupted disposable cache must recover from the unchanged
            # source, rather than breaking playback or modifying raw data.
            cache = next((repo / "outputs/viser/dataset_replay/clips").glob("*.npz"))
            cache.write_bytes(b"interrupted npz write")
            recovered = catalog.load(entry)
            np.testing.assert_array_equal(recovered["joint_pos"], expected["joint_pos"][3:])
            # Some future dataset releases may repeat a name across archives.
            # The chooser labels and canonical keys must still be unique.
            path.with_name("tripod_extra.npz").write_bytes(path.read_bytes())
            duplicated = DatasetCatalog(repo)
            self.assertEqual(len(duplicated.entries), 4)
            self.assertEqual(len(duplicated.by_label), 4)
            self.assertEqual(len(duplicated.by_key), 4)

    def test_revo2_requires_real_source_split_and_keeps_source_hands_for_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            folder = repo / "data/revo2/pilot_position"
            folder.mkdir(parents=True)
            source = visual_arrays(7)
            target = visual_arrays(7, 51, 83)
            target.update({f"source_{key}": source[key] for key in ("joint_pos", "body_pos_w", "body_quat_w")})
            np.savez_compressed(
                folder / "sub16.npz",
                **target,
                joint_names=[f"target_joint_{i}" for i in range(51)],
                body_names=[f"target_body_{i}" for i in range(83)],
                source_joint_names=[f"source_joint_{i}" for i in range(53)],
                source_body_names=[f"source_body_{i}" for i in range(54)],
                motion_names=["sub16_clothesstand_012_v00"],
                motion_lengths=[7],
                object_names=["clothesstand"],
                fps=[50],
            )
            clip = {
                "motion_name": "sub16_clothesstand_012_v00",
                "output": "data/revo2/pilot_position/sub16.npz",
                "source_split": "test",
                "train_ready": False,
            }
            manifest = folder / "manifest.json"
            manifest.write_text(json.dumps({"method": "position", "clips": [clip]}))
            catalog = DatasetCatalog(repo)
            self.assertEqual(len(catalog.search(hand="both", split="test")), 1)
            self.assertEqual(catalog.search(split="train"), [])
            data = catalog.load(catalog.entries[0])
            self.assertTrue(data["preview_only"][0])
            np.testing.assert_array_equal(data["source_joint_pos"], source["joint_pos"])
            del clip["source_split"]
            manifest.write_text(json.dumps({"method": "position", "clips": [clip]}))
            self.assertEqual(DatasetCatalog(repo).entries, ())


if __name__ == "__main__":
    unittest.main()
