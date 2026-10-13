"""Verify stage order, failure gating, and exclusion of specialist weights."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/reproduce/pickup_then_full.py"
spec = importlib.util.spec_from_file_location("pickup_chain", path)
chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chain)


class PickupPipelineTests(unittest.TestCase):
    def test_success_queues_original_paper_checkpoint_after_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper = root / "original-paper/model_300.pt"
            specialist = root / "specialist/model_2999.pt"
            events, calls = [], []

            def trainer(directory, stage, env, extra_args=()):
                events.append(stage); calls.append((env, list(extra_args)))
                return root / stage

            def evaluate(directory, checkpoint):
                self.assertEqual(checkpoint, specialist)
                events.append("evaluate")

            chain.run_chain(root, paper, trainer=trainer, evaluator=evaluate,
                            checkpoint_finder=lambda job: specialist)
            self.assertEqual(events, ["TRAINING_PICKUPS", "evaluate", "TRAINING_PAPER_FULL"])
            self.assertEqual(calls[0][0]["WEAVE_TRAIN_ITERATIONS"], "3000")
            self.assertEqual(calls[1][0]["WEAVE_TRAIN_ITERATIONS"], "99700")
            self.assertEqual(calls[1][0]["WEAVE_TRAIN_TASK"], "G1-Inspire-HOI-v0")
            self.assertEqual(calls[1][1], ["--resume", "--load_run", str(paper.parent),
                                         "--checkpoint", paper.name])
            self.assertNotIn(str(specialist), calls[1][1])

    def test_training_failure_does_not_launch_evaluation_or_full_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); events = []

            def fail(directory, stage, env, extra_args=()):
                events.append(stage)
                raise subprocess.CalledProcessError(1, "trainer")

            with self.assertRaises(subprocess.CalledProcessError):
                chain.run_chain(root, root / "model_300.pt", trainer=fail,
                                evaluator=lambda *args: events.append("evaluate"))
            self.assertEqual(events, ["TRAINING_PICKUPS"])

    def test_evaluation_error_does_not_launch_full_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); events = []

            def trainer(directory, stage, env, extra_args=()):
                events.append(stage); return root

            def fail_evaluation(*args):
                raise RuntimeError("missing evaluation report")

            with self.assertRaises(RuntimeError):
                chain.run_chain(root, root / "model_300.pt", trainer=trainer,
                                evaluator=fail_evaluation,
                                checkpoint_finder=lambda job: root / "model_2999.pt")
            self.assertEqual(events, ["TRAINING_PICKUPS"])


if __name__ == "__main__":
    unittest.main()
