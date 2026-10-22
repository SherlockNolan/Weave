#!/usr/bin/env python3
"""Open policy evaluation, or recorded dataset replay with --dataset, in Viser."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from record_pickups import REPO, extract_clip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sequences = ["sub1_tripod_075", "sub16_clothesstand_012", "sub9_clothesstand_039", "sub7_whitechair_072"]
    parser.add_argument("--sequence", choices=sequences, default=sequences[0])
    parser.add_argument("--variant", type=int, default=0, help="Variant index, starting at zero.")
    parser.add_argument("--checkpoint", type=Path, help="Defaults to the most recently saved unified paper policy.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--viser-host", default="127.0.0.1")
    parser.add_argument("--dataset", action="store_true", help="Replay recorded dataset trajectories on CPU.")
    parser.add_argument("--dataset-hand", choices=("inspire", "revo2", "both"), default="both")
    parser.add_argument("--dataset-split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--dataset-method", choices=("position", "vector"), default="position")
    parser.add_argument("--viser-no-wait", action="store_true")
    parser.add_argument("--viser-exit", action="store_true")
    parser.add_argument("--single-task", action="store_true", help="Disable the searchable task browser.")
    parser.add_argument("--viser-catalog", choices=("all", "train", "test", "pickups"), default="all")
    parser.add_argument(
        "--original-scene",
        action="store_true",
        help="Use the original training scene without the fitted chair support.",
    )
    args = parser.parse_args()
    os.chdir(REPO)
    if args.dataset:
        command = [
            sys.executable,
            "-u",
            str(REPO / "scripts/reproduce/viser_dataset.py"),
            "--host",
            args.viser_host,
            "--port",
            str(args.viser_port),
            "--hand",
            args.dataset_hand,
            "--split",
            args.dataset_split,
            "--method",
            args.dataset_method,
        ]
        # All clips remain selectable in the browser, independent of the initial pickup.
        os.execv(sys.executable, command)
    clips = json.loads((REPO / "data/pickups/manifest.json").read_text())["clips"]
    clips = sorted([clip for clip in clips if clip["sequence"] == args.sequence], key=lambda c: c["motion_name"])
    if not 0 <= args.variant < len(clips):
        parser.error(f"{args.sequence} has variants 0..{len(clips) - 1}")
    if args.checkpoint is None:
        candidates = [
            p
            for p in (REPO / "logs/rsl_rl/g1_inspire_hoi").glob("*paper-reference*/model_*.pt")
            if p.stat().st_size > 0 and time.time() - p.stat().st_mtime > 5
        ]
        if not candidates:
            parser.error("No unified policy checkpoint found; pass --checkpoint")
        args.checkpoint = max(candidates, key=lambda p: p.stat().st_mtime)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    clip = clips[args.variant]
    output = REPO / "outputs/viser" / checkpoint.stem / clip["motion_name"]
    motion = REPO / "outputs/viser/inputs" / (clip["motion_name"] + ".npz")
    _, lookat = extract_clip(clip["object"], clip["motion_name"], motion)
    eye = [lookat[0] + 4.5, lookat[1] - 4.5, 3.0]
    command = [
        sys.executable,
        "-u",
        "scripts/rsl_rl/eval.py",
        "--viser",
        "--headless",
        "--task",
        "G1-Inspire-HOI-v0" if args.original_scene else "G1-Inspire-HOI-Pickup-v0",
        "--config-dir",
        "./configs/track",
        "--config-name",
        "eval" if args.original_scene else "eval_pickups",
        "--checkpoint",
        str(checkpoint),
        "--num_envs",
        "1",
        "--seed",
        "42",
        "--device",
        args.device,
        "--viser-port",
        str(args.viser_port),
        "--viser-host",
        args.viser_host,
        "--output_dir",
        str(output if args.single_task else REPO / "outputs/viser" / checkpoint.stem / "tasks"),
        f"env.commands.motion.motion_files=[{motion}]",
        "env.viewer.origin_type=env",
        f"env.viewer.eye={eye}",
        f"env.viewer.lookat={lookat}",
        "--kit_args=--/plugins/carb.tasking.plugin/threadCount=4 --/plugins/omni.tbb.globalcontrol/maxThreadCount=4",
    ]
    if not args.single_task:
        command += ["--viser-tasks", "--viser-catalog", args.viser_catalog]
    for flag in ("viser_no_wait", "viser_exit"):
        if getattr(args, flag):
            command.append("--" + flag.replace("_", "-"))
    print(f"[VISER] Checkpoint: {checkpoint}\n[VISER] Reference: {clip['motion_name']}", flush=True)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
