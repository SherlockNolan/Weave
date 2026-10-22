"""Regression checks for clone alignment, contact semantics, and optimizer restoration."""

import copy
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1] / "source/g1_hoi_learning/g1_hoi_learning"


def load_module(name, relative_path):
    # Loading these independent modules directly avoids starting Kit in CPU tests.
    spec = importlib.util.spec_from_file_location(name, SOURCE / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


layout = load_module("weave_test_layout", "scene_layout.py")
compat = load_module("weave_test_compat", "isaac_compat.py")
optimizers = load_module("weave_test_optimizers", "algorithms/optimizers.py")


class SceneLayoutTests(unittest.TestCase):
    def test_every_training_environment_matches_its_reference(self):
        for count in (1, 2, 9, 19, 2048):
            with self.subTest(count=count):
                actual = layout.round_robin_clone_strategy(np.ones(9), count)
                np.testing.assert_array_equal(actual, np.arange(count) % 9)

    def test_support_table_and_object_are_paired_for_all_environments(self):
        for objects in (1, 3, 9):
            with self.subTest(objects=objects):
                prototypes = np.array([(a, b) for a in range(objects) for b in range(objects)])
                actual = prototypes[layout.paired_round_robin_clone_strategy(np.ones(objects**2), 2048)]
                np.testing.assert_array_equal(actual[:, 0], np.arange(2048) % objects)
                np.testing.assert_array_equal(actual[:, 1], actual[:, 0])

    def test_incompatible_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            layout.round_robin_clone_strategy(np.array([1, 2]), 18)
        with self.assertRaises(ValueError):
            layout.paired_round_robin_clone_strategy(np.ones(3), 18)

    def test_guard_checks_environments_beyond_the_first_object_cycle(self):
        def prim(source):
            return SimpleNamespace(
                IsValid=lambda: True,
                GetPrimStack=lambda: [SimpleNamespace(layer=SimpleNamespace(realPath=source))],
            )

        sources = {"a": "/assets/a.usd", "b": "/assets/b.usd"}
        names = ["a", "b", "a", "b"]
        prims = {f"/World/envs/env_{i}/Object": prim(sources[n]) for i, n in enumerate(names)}
        stage = SimpleNamespace(GetPrimAtPath=prims.__getitem__)
        self.assertEqual(layout.validate_object_assignments(stage, names, sources), {"a": 2, "b": 2})
        prims["/World/envs/env_3/Object"] = prim(sources["a"])
        with self.assertRaisesRegex(RuntimeError, "env 3: reference=b"):
            layout.validate_object_assignments(stage, names, sources)


class ContactSemanticsTests(unittest.TestCase):
    def test_lab3_friction_does_not_change_normal_contact_flags(self):
        normal = torch.tensor([[[[0.0, 0.0, 0.5]]]])
        data = SimpleNamespace(normal_force_matrix_w=SimpleNamespace(torch=normal), force_matrix_w=normal + 100)
        with patch.object(compat, "ISAACLAB3", True):
            actual = compat.normal_contact_force_matrix(SimpleNamespace(data=data))
        self.assertIs(actual, normal)
        self.assertFalse(bool((actual.norm(dim=-1) > 1.0).any()))

    def test_lab3_feet_history_uses_normal_forces(self):
        normal = torch.zeros(2, 3, 2, 3)
        data = SimpleNamespace(
            net_normal_forces_w_history=SimpleNamespace(torch=normal), net_forces_w_history=normal + 100,
        )
        with patch.object(compat, "ISAACLAB3", True):
            self.assertIs(compat.net_normal_contact_forces(SimpleNamespace(data=data), history=True), normal)

    def test_lab2_preserves_original_tensor_interfaces(self):
        normal = torch.ones(2, 1, 54, 3)
        history = torch.ones(2, 3, 2, 3)
        sensor = SimpleNamespace(data=SimpleNamespace(force_matrix_w=normal, net_forces_w_history=history))
        with patch.object(compat, "ISAACLAB3", False):
            self.assertIs(compat.normal_contact_force_matrix(sensor), normal)
            self.assertIs(compat.net_normal_contact_forces(sensor, history=True), history)


class OptimizerResumeTests(unittest.TestCase):
    def test_restored_momentum_and_adaptive_rate_match_uninterrupted_updates(self):
        torch.manual_seed(7)
        model = torch.nn.Linear(4, 3)
        optimizer = optimizers.MuonAdamWWrapper([model], lr=1.3e-4)
        inputs = torch.randn(5, 4)

        def step(net, opt):
            opt.zero_grad()
            net(inputs).square().mean().backward()
            opt.step()

        step(model, optimizer)
        resumed = copy.deepcopy(model)
        restored = optimizers.MuonAdamWWrapper([resumed], lr=0.01)
        restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        algorithm = SimpleNamespace(optimizer=restored, learning_rate=0.01)
        self.assertEqual(optimizers.restore_learning_rate(algorithm), 1.3e-4)
        self.assertEqual(algorithm.learning_rate, 1.3e-4)
        for opt in (optimizer, restored):
            # Exercise both RSL's public parameter-group interface and the direct setter.
            for group in opt.param_groups:
                group["lr"] = 2e-5
            self.assertTrue(all(g["lr"] == 2e-5 for child in opt.optimizers for g in child.param_groups))
            opt.set_learning_rate(3e-5)
        step(model, optimizer)
        step(resumed, restored)
        for expected, actual in zip(model.parameters(), resumed.parameters()):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_conflicting_restored_rates_are_rejected(self):
        parameters = [torch.nn.Parameter(torch.ones(1)) for _ in range(2)]
        children = [torch.optim.AdamW([p], lr=rate) for p, rate in zip(parameters, [1e-4, 2e-4])]
        algorithm = SimpleNamespace(optimizer=optimizers.OptimizerGroup(children), learning_rate=1e-3)
        with self.assertRaisesRegex(ValueError, "one shared restored learning rate"):
            optimizers.restore_learning_rate(algorithm)


if __name__ == "__main__":
    unittest.main()
