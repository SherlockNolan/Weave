"""CPU regressions for reference ordering, held-out identity and hand constraints.

These tests never import Isaac or allocate a CUDA device. The small conversion
test supplies a deterministic optimizer so it can verify archive semantics
without conflating them with nonlinear optimization or physical grasp quality.
"""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/reproduce"))

import retarget_revo2 as pipeline
from retarget_dataset_io import FRAME_FIELDS, get_source_layout, load_clip
from revo2_model import UrdfModel


class SourceArchiveTests(unittest.TestCase):
    def test_layout_matches_real_simulator_snapshot_not_urdf_xml_order(self):
        snapshot = REPO / "outputs/code-audit/2026-10-19/scene-old.json"
        if not snapshot.is_file():
            self.skipTest("Original simulator audit is not available")
        runtime = json.loads(snapshot.read_text())
        layout = get_source_layout()
        self.assertEqual(layout["joint_names"], runtime["joint_names"])
        self.assertEqual(layout["body_names"], runtime["body_names"])
        self.assertEqual(layout["fixed_link_to_body"]["L_hand_base_link"], "left_wrist_yaw_link")
        self.assertEqual(layout["fixed_link_to_body"]["L_index_tip"], "L_index_intermediate")

    def test_whole_clips_and_original_train_test_identity_are_retained(self):
        expected = (
            ("tripod", "sub1_tripod_075", 417, "train"),
            ("clothesstand", "sub16_clothesstand_012", 354, "test"),
            ("clothesstand", "sub9_clothesstand_039", 467, "train"),
            ("whitechair", "sub7_whitechair_072", 207, "train"),
        )
        for obj, sequence, count, split in expected:
            with self.subTest(sequence=sequence):
                path = REPO / f"data/pickups/{obj}_pickups.npz"
                if not path.is_file():
                    self.skipTest("Original pickup subsets are not available")
                actual, metadata = load_clip(path, sequence)
                self.assertEqual(metadata["frames"], count)
                self.assertEqual(metadata["fps"], 50)
                self.assertEqual(metadata["source_split"], split)
                self.assertIn(f"data/{split}/", metadata["original_source_path"])
                self.assertEqual(actual["motion_lengths"].tolist(), [count])
                self.assertEqual(actual["joint_pos"].shape, (count, 53))
                self.assertEqual(actual["body_pos_w"].shape, (count, 54, 3))
                # Compare every preserved channel against the independently
                # indexed source archive, including its final frame.
                with np.load(path, allow_pickle=False) as archive:
                    index = list(archive["motion_names"].astype(str)).index(metadata["motion_name"])
                    starts = np.r_[0, np.cumsum(archive["motion_lengths"])]
                    for field in FRAME_FIELDS:
                        np.testing.assert_array_equal(actual[field], archive[field][starts[index] : starts[index + 1]])

    def test_unavailable_variant_does_not_silently_reuse_whitechair_clip(self):
        path = REPO / "data/pickups/whitechair_pickups.npz"
        if not path.is_file():
            self.skipTest("Original pickup subsets are not available")
        with self.assertRaisesRegex(ValueError, "absent"):
            load_clip(path, "sub7_whitechair_072", variant=1)


class HandConstraintTests(unittest.TestCase):
    def test_projection_respects_real_active_and_dependent_joint_bounds(self):
        metadata = json.loads((pipeline.TARGET / "assembly.json").read_text())
        model = UrdfModel(pipeline.ASSEMBLED_URDF)
        names = metadata["hand_drive_joints"]
        # Deliberately infeasible position jumps make both position and speed
        # constraints binding; retaining the original timing is required.
        raw = np.random.default_rng(681).normal(0.8, 1.2, (207, len(names)))
        before = raw.copy()
        controls = pipeline.rate_project(raw, names, metadata, fps=50)
        self.assertEqual(controls.shape, raw.shape)
        np.testing.assert_array_equal(raw, before)
        self.assertFalse(np.allclose(controls, raw))
        resolved = [model.configuration(dict(zip(names, row))) for row in controls]
        for name, bounds in metadata["hand_joint_limits"].items():
            with self.subTest(joint=name):
                angles = np.asarray([state[name] for state in resolved])
                self.assertGreaterEqual(angles.min(), bounds["lower"] - 1e-10)
                self.assertLessEqual(angles.max(), bounds["upper"] + 1e-10)
                self.assertLessEqual(np.abs(np.diff(angles)).max() * 50, bounds["velocity"] + 1e-10)


class ConvertedArchiveTests(unittest.TestCase):
    def test_actual_export_preserves_labels_body_states_and_held_out_identity(self):
        source_path = REPO / "data/pickups/clothesstand_pickups.npz"
        if not source_path.is_file():
            self.skipTest("Original pickup subsets are not available")
        source, info = load_clip(source_path, "sub16_clothesstand_012")
        count = 5
        for key in FRAME_FIELDS:
            source[key] = source[key][:count].copy()
        source["motion_lengths"] = np.asarray([count], dtype=np.int64)
        info.update(frames=count, duration_seconds=count / info["fps"])
        layout = get_source_layout()
        labels = source["contact_label"]
        labels[:] = 0
        # Two old thumb segments collapse to one new distal segment. Positive
        # contact wins, both negative mean avoid, neutral remains unconstrained.
        combinations = [(1, -1), (-1, -1), (-1, 0), (0, 0), (0, 1)]
        for frame, pair in enumerate(combinations):
            labels[frame, layout["body_names"].index("L_thumb_intermediate")] = pair[0]
            labels[frame, layout["body_names"].index("L_thumb_distal")] = pair[1]
        labels[:, layout["body_names"].index("pelvis")] = -1
        labels[:, layout["body_names"].index("left_wrist_yaw_link")] = 1

        metadata = json.loads((pipeline.TARGET / "assembly.json").read_text())
        model = UrdfModel(pipeline.ASSEMBLED_URDF)
        active = metadata["hand_drive_joints"]
        fixed = metadata["body_drive_joints"]
        target_names = [name for name, joint in model.joints.items() if joint.attrib["type"] != "fixed"]

        class FakeRetarget:
            joint_names = target_names
            joint_limits = np.asarray(
                [[metadata["hand_joint_limits"][name][key] for key in ("lower", "upper")] for name in active]
            )
            optimizer = SimpleNamespace(
                target_joint_names=active,
                fixed_joint_names=fixed,
                opt=SimpleNamespace(last_optimize_result=lambda: 1),
            )

            def retarget(self, desired, fixed_qpos):
                values = dict(zip(fixed, fixed_qpos))
                values.update(dict(zip(active, self.joint_limits.mean(axis=1))))
                resolved = model.configuration(values)
                return np.asarray([resolved[name] for name in self.joint_names])

        with tempfile.TemporaryDirectory(prefix="weave-retarget-test-") as temporary:
            args = SimpleNamespace(
                method="position", variant=0, max_frames=0, output=Path(temporary), contact_refine=False
            )
            with (
                patch.object(pipeline, "load_clip", return_value=(source, info)),
                patch.object(pipeline, "optimizer", return_value=FakeRetarget()),
                redirect_stdout(io.StringIO()),
            ):
                report = pipeline.convert_clip(args, "sub16_clothesstand_012")
            self.assertEqual(report["source_split"], "test")
            self.assertFalse(report["train_ready"])
            with np.load(report["output"], allow_pickle=False) as result:
                self.assertTrue(bool(result["preview_only"]))
                self.assertEqual(result["motion_lengths"].tolist(), [count])
                target_bodies = list(result["body_names"].astype(str))
                target_joints = list(result["joint_names"].astype(str))
                for name in ("left_thumb_distal_link", "left_thumb_tip_link", "left_thumb_touch_link"):
                    np.testing.assert_array_equal(
                        result["contact_label"][:, target_bodies.index(name)], [1, -1, 0, 0, 1]
                    )
                np.testing.assert_array_equal(
                    result["contact_label"][:, target_bodies.index("pelvis_contour_link")], -np.ones(count)
                )
                np.testing.assert_array_equal(
                    result["contact_label"][:, target_bodies.index("left_hand_base_link")], np.ones(count)
                )
                for name in metadata["body_drive_joints"]:
                    original_index = layout["joint_names"].index(name)
                    target_index = target_joints.index(name)
                    np.testing.assert_array_equal(
                        result["joint_pos"][:, target_index], source["joint_pos"][:, original_index]
                    )
                    np.testing.assert_array_equal(
                        result["joint_vel"][:, target_index], source["joint_vel"][:, original_index]
                    )
                for name in layout["body_names"][:30]:
                    np.testing.assert_allclose(
                        result["body_pos_w"][:, target_bodies.index(name)],
                        source["body_pos_w"][:, layout["body_names"].index(name)],
                        atol=2e-6,
                        rtol=0,
                    )
                for key in FRAME_FIELDS:
                    if key.startswith("object_"):
                        np.testing.assert_array_equal(result[key], source[key])
                    else:
                        np.testing.assert_array_equal(result["source_" + key], source[key])


if __name__ == "__main__":
    unittest.main()
