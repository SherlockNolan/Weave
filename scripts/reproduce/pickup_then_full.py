#!/usr/bin/env python3
"""Run pickup-specialist training/evaluation, then resume the clean paper run."""
from __future__ import annotations

import argparse
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
GPU_UUIDS = (
    "GPU-6de172b1-a4eb-dd56-d8ae-00be059f483f",
    "GPU-686bfbdb-b04d-6585-44ed-033046713f67",
)


def write_state(directory: Path, phase: str, **details):
    state = {"phase": phase, "updated_at": datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=8))).isoformat(), **details}
    temp = directory / "state.json.tmp"
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(directory / "state.json")
    print(json.dumps(state), flush=True)


def wait_for_idle_gpus(directory: Path, phase: str, indices=(0, 1)):
    while True:
        raw = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True)
        apps = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
        busy = []
        for row in csv.reader(raw.splitlines()):
            index = int(row[0])
            if index not in indices:
                continue
            assert row[1].strip() == GPU_UUIDS[index], "Physical GPU assignment changed"
            if int(row[2]) > 64 or int(row[3]) > 0 or row[1].strip() in apps.splitlines():
                busy.append(index)
        if not busy:
            return
        write_state(directory, "WAITING_FOR_IDLE_GPUS", next_phase=phase, busy_gpus=busy)
        time.sleep(15)


def train_stage(directory: Path, stage: str, env_updates: dict, extra_args=(),
                runner=subprocess.run, waiter=wait_for_idle_gpus) -> Path:
    waiter(directory, stage)
    write_state(directory, stage)
    pointer = directory / f"{stage.lower()}.jobdir"
    env = dict(os.environ, **env_updates, WEAVE_JOB_PTR=str(pointer))
    runner(["bash", "scripts/reproduce/train_gpu01.sh", *extra_args],
           cwd=REPO, env=env, check=True)
    job = Path(pointer.read_text().strip())
    if (job / "status").read_text().strip() != "COMPLETED":
        raise RuntimeError(f"Training stage did not complete: {job}")
    return job


def latest_checkpoint(job: Path) -> Path:
    meta = json.loads((job / "launch.json").read_text())
    paths = list(Path(meta["checkpoint_dir"]).glob("model_*.pt"))
    if not paths:
        raise RuntimeError(f"No checkpoint saved for {job}")
    return max(paths, key=lambda path: int(path.stem.split("_")[-1]))


def evaluate_pickups(directory: Path, checkpoint: Path, runner=subprocess.run,
                     waiter=wait_for_idle_gpus):
    reports = {}
    for obj in ("tripod", "clothesstand", "whitechair"):
        waiter(directory, "EVALUATING_PICKUPS", indices=(0,))
        write_state(directory, "EVALUATING_PICKUPS", object=obj, checkpoint=str(checkpoint))
        output = directory / "eval" / obj
        output.mkdir(parents=True, exist_ok=True)
        command = ["bash", "scripts/reproduce/run_gpu0.sh", "scripts/rsl_rl/eval.py",
                   "--task", "G1-Inspire-HOI-Pickup-v0", "--headless",
                   "--config-dir", "./configs/track", "--config-name", "eval_pickups",
                   "--checkpoint", str(checkpoint), "--output_dir", str(output),
                   f"env.commands.motion.motion_files=[./data/pickups/{obj}_pickups.npz]"]
        with (output / "eval.log").open("w") as stream:
            runner(command, cwd=REPO, check=True, stdout=stream, stderr=subprocess.STDOUT)
        report = json.loads((output / "metrics.json").read_text())
        expected = {"tripod": 3, "clothesstand": 8, "whitechair": 1}[obj]
        if report["num_clips"] != expected:
            raise RuntimeError(f"Incomplete evaluation for {obj}")
        reports[obj] = {"clips": expected, "success_rate": report["success_rate"],
                        "progress_rate": report["progress_rate"],
                        "report": str(output / "metrics.json")}
    (directory / "pickup-evaluation.json").write_text(json.dumps(reports, indent=2) + "\n")
    return reports


def run_chain(directory: Path, paper_checkpoint: Path, specialist_iterations: int = 3000,
              trainer=train_stage, evaluator=evaluate_pickups, checkpoint_finder=latest_checkpoint):
    # The paper stage deliberately resumes its original train-only checkpoint.
    # Pickup-specialist weights include a published test clip and must not seed it.
    paper_iteration = int(paper_checkpoint.stem.split("_")[-1])
    remaining = 100000 - paper_iteration
    if remaining <= 0:
        raise ValueError("The paper checkpoint already reached the target budget")
    plan = {"specialist_iterations": specialist_iterations, "specialist_total_clips": 12,
            "paper_checkpoint": str(paper_checkpoint), "paper_resume_iteration": paper_iteration,
            "paper_additional_iterations": remaining, "paper_total_target": 100000,
            "paper_uses_specialist_weights": False}
    (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    pickup_job = trainer(directory, "TRAINING_PICKUPS", {
        "WEAVE_TRAIN_TASK": "G1-Inspire-HOI-Pickup-v0",
        "WEAVE_TRAIN_CONFIG": "train_pickups_dual",
        "WEAVE_TRAIN_RUN_NAME": "pickups-specialist-dual",
        "WEAVE_TRAIN_ITERATIONS": str(specialist_iterations),
        "WEAVE_TRAIN_ENVS_PER_GPU": "2048",
    })
    checkpoint = checkpoint_finder(pickup_job)
    (directory / "pickup-checkpoint.txt").write_text(str(checkpoint) + "\n")
    evaluator(directory, checkpoint)
    paper_job = trainer(directory, "TRAINING_PAPER_FULL", {
        "WEAVE_TRAIN_TASK": "G1-Inspire-HOI-v0",
        "WEAVE_TRAIN_CONFIG": "train_reference_dual",
        "WEAVE_TRAIN_RUN_NAME": "paper-reference-resumed",
        "WEAVE_TRAIN_ITERATIONS": str(remaining),
        "WEAVE_TRAIN_ENVS_PER_GPU": "2048",
    }, ["--resume", "--load_run", str(paper_checkpoint.parent),
        "--checkpoint", paper_checkpoint.name])
    write_state(directory, "COMPLETED", pickup_job=str(pickup_job), paper_job=str(paper_job))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-checkpoint", type=Path, required=True)
    parser.add_argument("--specialist-iterations", type=int, default=3000)
    args = parser.parse_args()
    if not args.paper_checkpoint.is_file():
        parser.error("The preserved paper checkpoint is missing")
    if args.specialist_iterations <= 0:
        parser.error("specialist-iterations must be positive")
    os.chdir(REPO)
    manifest = json.loads((REPO / "data/pickups/manifest.json").read_text())
    assert manifest["total_clips"] == 12
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d_%H-%M-%S")
    directory = REPO / "logs/training" / (stamp + "_pickups-then-paper")
    with (REPO / "logs/preparation/pickup-paper-chain.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory.mkdir(parents=True)
        link = REPO / "logs/training/latest-chain"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(directory.name)
        try:
            run_chain(directory, args.paper_checkpoint.resolve(), args.specialist_iterations)
        except KeyboardInterrupt:
            write_state(directory, "STOPPED_BY_USER")
            raise
        except Exception as exc:
            write_state(directory, "FAILED", error=str(exc), paper_training_not_auto_started_after_failure=True)
            raise


if __name__ == "__main__":
    main()
