#!/usr/bin/env python3
"""Evaluate the final unified checkpoint on both full reference splits."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PAPER = "https://arxiv.org/html/2609.16683v1#S5.SS2"
TARGETS = {
    "train": {"success_rate": 0.925, "progress_rate": 0.963},
    "test": {"success_rate": 0.650, "progress_rate": 0.845},
}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def gpu_state():
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], text=True
    )
    processes = list(csv.reader(apps.splitlines(), skipinitialspace=True))
    return [
        {
            "index": int(i),
            "uuid": u.strip(),
            "memory_mib": int(m),
            "free_mib": int(f),
            "total_mib": int(t),
            "utilization": int(p),
            "processes": [r for r in processes if r[0].strip() == u.strip()],
        }
        for i, u, m, f, t, p in csv.reader(text.splitlines(), skipinitialspace=True)
    ]


def select_gpu(state, indices, share, minimum_free_gib):
    """Respect an explicit shared-GPU request while preserving memory headroom."""
    candidates = [
        g
        for g in state
        if g["index"] in indices
        and (
            g["free_mib"] >= minimum_free_gib * 1024
            if share
            else g["memory_mib"] <= 128 and g["utilization"] == 0 and not g["processes"]
        )
    ]
    return max(candidates, key=lambda g: (g["free_mib"], -g["utilization"]), default=None)


def mean(values):
    return sum(values) / len(values) if values else None


def aggregate(clips, metric_keys):
    return {
        "num_clips": len(clips),
        "num_success": sum(c["success"] for c in clips),
        "success_rate": mean([float(c["success"]) for c in clips]),
        "progress_rate": mean([c["progress"] for c in clips]),
        "failures": dict(Counter(c["cause"] for c in clips if not c["success"])),
        "metrics_clip_balanced": {k: mean([c[k] for c in clips]) for k in metric_keys},
        "metrics_frame_weighted": {
            k: sum(c[k] * c["steps"] for c in clips) / sum(c["steps"] for c in clips) for k in metric_keys
        },
    }


def summarize(output, plan, records):
    result = {
        "checkpoint": plan["checkpoint"],
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "device": plan["device"],
        "seed": plan["seed"],
        "updated_at": now(),
        "paper": PAPER,
        "complete": len(records) == len(plan["datasets"]),
        "splits": {},
    }
    lines = [
        "# 统一策略完整参考集评估",
        "",
        f"- 权重：`{plan['checkpoint']}`。",
        f"- SHA256：`{plan['checkpoint_sha256']}`。",
        f"- 设备：`{plan['device']}`；seed：{plan['seed']}；Isaac Sim：{plan['versions']['isaacsim']}。",
        f"- 状态：{'全部完成' if result['complete'] else '运行中，以下是已完成部分'}；更新时间：{now()}。",
        f"- [论文对照实验]({PAPER})：训练集 SR 92.5%、PR 96.3%；未见测试集 SR 65.0%、PR 84.5%。",
        "",
    ]
    if plan["device"] == "cpu":
        lines += [
            "本轮是 CPU PhysX 诊断评估。GPU PhysX 与 CPU PhysX 可能产生不同接触轨迹，"
            "不能标作论文原 GPU 实验的严格复现。",
            "",
        ]
    lines += [
        "本地训练在中途由 Isaac Sim 5.1 / Isaac Lab 2.3 迁移到 6.2 / 3.0，"
        "因此即使评估协议对齐，也存在训练与仿真版本差异。",
        "",
        "成功要求运行至参考片段末尾且未触发任何失败项；完成与失败同时发生按失败。",
        "保留官方旧统计以核对该边界，失败阈值与随机化保持 eval_reference 原值。",
        "每片段从第 0 帧开始；长参考的回合上限延长到允许完整结束；使用原始无支撑桌场景。",
        "SR/PR 按片段汇总，跟踪误差同时提供论文的片段平均和原代码的帧加权平均。",
        "",
        "| Split | 物体 | 成功 / 全部 | SR | PR |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for entry in plan["datasets"]:
        key = (entry["split"], entry["object"])
        if key not in records:
            continue
        r = records[key]
        successes = sum(c["success"] for c in r["clips"])
        lines.append(
            f"| {key[0]} | {key[1]} | {successes}/{r['num_clips']} | "
            f"{100 * successes / r['num_clips']:.2f}% | {100 * r['progress_rate']:.2f}% |"
        )
    lines += [
        "",
        "| Split | 已评估片段 | SR | 论文 SR | 相差百分点 | PR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in plan["splits"]:
        selected = [r for (s, _), r in records.items() if s == split]
        if not selected:
            continue
        clips = [c for r in selected for c in r["clips"]]
        data = aggregate(clips, selected[0]["metric_keys"])
        expected = sum(e["num_clips"] for e in plan["datasets"] if e["split"] == split)
        data["expected_clips"] = expected
        data["complete"] = len(clips) == expected
        data["simultaneous_clip_end_and_failure"] = sum(r["simultaneous_clip_end_and_failure"] for r in selected)
        data["success_rate_official_legacy"] = (data["num_success"] + data["simultaneous_clip_end_and_failure"]) / len(
            clips
        )
        data["paper_target"] = TARGETS[split]
        data["success_gap_percentage_points"] = 100 * (data["success_rate"] - TARGETS[split]["success_rate"])
        result["splits"][split] = data
        lines.append(
            f"| {split} | {len(clips)}/{expected} | {100 * data['success_rate']:.2f}% | "
            f"{100 * TARGETS[split]['success_rate']:.1f}% | {data['success_gap_percentage_points']:+.2f} | "
            f"{100 * data['progress_rate']:.2f}% |"
        )
    lines += [
        "",
        "逐片段结果与终止原因保存在各物体目录的 `metrics.json`；`plan.json` 记录版本、数据清单与运行设置。",
        "",
    ]
    write_json(output / "summary.json", result)
    (output / "README.md").write_text("\n".join(lines))


def load_result(path, entry, checkpoint, seed, device):
    r = json.loads(path.read_text())
    if (
        Path(r["checkpoint"]).resolve() != checkpoint
        or r["num_clips"] != entry["num_clips"]
        or r.get("num_finished") != entry["num_clips"]
        or [c["name"] for c in r["clips"]] != entry["names"]
        or not all(c["finished"] for c in r["clips"])
    ):
        raise RuntimeError(f"Incomplete or mismatched evaluation: {path}")
    settings = r["evaluation_settings"]
    if (
        settings["seed"] != seed
        or settings["device"] != device
        or settings["episode_length_s"] < entry["max_frames"] / entry["fps"]
    ):
        raise RuntimeError(f"Evaluation settings differ from the plan: {path}")
    return r


def eval_command(entry, checkpoint, seed, device, directory, threads):
    return [
        sys.executable,
        "-u",
        "scripts/rsl_rl/eval.py",
        "--headless",
        "--task",
        "G1-Inspire-HOI-v0",
        "--config-dir",
        "./configs/track",
        "--config-name",
        "eval_reference",
        "--checkpoint",
        str(checkpoint),
        "--seed",
        str(seed),
        "--device",
        device,
        "--output_dir",
        str(directory),
        f"env.commands.motion.motion_files=[{entry['path']}]",
        f"env.episode_length_s={entry['episode_length_s']}",
        "--kit_args=--/renderer/multiGpu/enabled=false --/renderer/multiGpu/maxGpuCount=1 "
        f"--/plugins/carb.tasking.plugin/threadCount={threads} "
        f"--/plugins/omni.tbb.globalcontrol/maxThreadCount={threads}",
    ]


def parallel_cpu(args, plan, output, checkpoint, env, records):
    """Run independent object evaluations, with one owner of summary writes."""
    stop = threading.Event()
    lock = threading.RLock()
    active = {}
    selected = next(g for g in gpu_state() if g["index"] == args.cpu_pinned_gpu)
    env = dict(env, CUDA_VISIBLE_DEVICES=selected["uuid"])
    pending = []
    for entry in plan["datasets"]:
        directory = output / entry["split"] / entry["object"]
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "metrics.json").exists():
            records[(entry["split"], entry["object"])] = load_result(
                directory / "metrics.json", entry, checkpoint, args.seed, "cpu"
            )
        else:
            pending.append(entry)
    summarize(output, plan, records)

    def update_status():
        with lock:
            write_json(
                output / "status.json",
                {
                    "state": "RUNNING",
                    "updated_at": now(),
                    "device": "cpu",
                    "workers": args.workers,
                    "active": dict(active),
                    "completed_objects": len(records),
                    "total_objects": len(plan["datasets"]),
                },
            )

    def work(entry):
        if stop.is_set():
            raise InterruptedError("Evaluation cancelled")
        directory = output / entry["split"] / entry["object"]
        key = f"{entry['split']}/{entry['object']}"
        command = eval_command(entry, checkpoint, args.seed, "cpu", directory, args.threads)
        write_json(directory / "command.json", {"command": command, "gpu": selected, "started_at": now()})
        print(f"[{now()}] Evaluating {key}: {entry['num_clips']} clips", flush=True)
        with (directory / "eval.log").open("w") as log:
            process = subprocess.Popen(
                command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            with lock:
                active[key] = {"child_pid": process.pid, "num_clips": entry["num_clips"]}
                update_status()
            try:
                while process.poll() is None:
                    if stop.wait(0.5):
                        os.killpg(process.pid, signal.SIGINT)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise InterruptedError("Evaluation cancelled")
                if process.returncode:
                    raise RuntimeError(f"Evaluation exited with {process.returncode}; see {directory / 'eval.log'}")
            finally:
                with lock:
                    active.pop(key, None)
                    update_status()
        return entry, load_result(directory / "metrics.json", entry, checkpoint, args.seed, "cpu")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, entry) for entry in pending]
        try:
            for future in as_completed(futures):
                entry, result = future.result()
                with lock:
                    records[(entry["split"], entry["object"])] = result
                    summarize(output, plan, records)
                    update_status()
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu"),
        default="auto",
        help="auto waits for an idle GPU; cpu is a separately labelled diagnostic.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1, help="Parallel CPU object evaluations (default: 1).")
    parser.add_argument("--splits", nargs="+", choices=("train", "test"), default=["test", "train"])
    parser.add_argument("--gpu-indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--share-gpu", action="store_true", help="Allow evaluation alongside existing GPU workloads.")
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=16,
        help="Minimum free GPU memory before a shared evaluation starts (default: 16 GiB).",
    )
    parser.add_argument(
        "--cpu-pinned-gpu",
        type=int,
        default=0,
        help="CUDA device used only for CPU pinned-memory allocation required by Isaac Lab.",
    )
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 3 or args.threads < 1:
        parser.error("Use 1..3 workers and a positive thread count")
    if args.workers > 1 and args.device != "cpu":
        parser.error("Multiple workers are only supported for CPU diagnostics")
    if args.share_gpu and args.device == "cpu":
        parser.error("--share-gpu selects shared GPU evaluation, not CPU diagnostics")
    if args.min_free_gib <= 0:
        parser.error("--min-free-gib must be positive")
    os.chdir(REPO)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Missing checkpoint: {checkpoint}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.nice(19)
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, allowed[-min(8 * args.workers, len(allowed)) :])
    datasets = []
    for split in args.splits:
        for path in sorted((REPO / "data" / split).glob("*.npz")):
            with np.load(path, allow_pickle=False) as z:
                lengths = z["motion_lengths"]
                fps = int(z["fps"].reshape(-1)[0])
                datasets.append(
                    {
                        "split": split,
                        "object": path.stem.split("_")[0],
                        "path": str(path),
                        "num_clips": len(lengths),
                        "fps": fps,
                        "max_frames": int(lengths.max()),
                        "episode_length_s": max(20.0, (int(lengths.max()) + 5) / fps),
                        "names": z["motion_names"].astype(str).tolist(),
                    }
                )
    for split in args.splits:
        if sum(e["num_clips"] for e in datasets if e["split"] == split) != {"train": 7869, "test": 1605}[split]:
            raise RuntimeError(f"The {split} dataset is incomplete")
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    versions = {p: version(p) for p in ("isaacsim", "isaaclab", "torch", "rsl-rl-lib", "numpy")}
    device = "cpu" if args.device == "cpu" else "cuda:0"
    plan = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "seed": args.seed,
        "device": device,
        "splits": args.splits,
        "versions": versions,
        "gpu_baseline": gpu_state(),
        "created_at": now(),
        "datasets": datasets,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "threads": args.threads,
        "workers": args.workers,
        "share_gpu": args.share_gpu,
        "minimum_free_gib": args.min_free_gib,
        "paper_targets": TARGETS,
    }
    old = output / "plan.json"
    if old.exists():
        previous = json.loads(old.read_text())
        for key in ("checkpoint_sha256", "seed", "device", "versions", "datasets"):
            if previous[key] != plan[key]:
                raise RuntimeError(f"Existing output has a different {key}; choose another --output")
        plan["created_at"] = previous["created_at"]
    write_json(old, plan)
    records = {}
    env = dict(
        os.environ,
        OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS=str(args.threads),
        HYDRA_FULL_ERROR="1",
    )
    try:
        if args.workers > 1:
            parallel_cpu(args, plan, output, checkpoint, env, records)
        for entry in [] if args.workers > 1 else datasets:
            directory = output / entry["split"] / entry["object"]
            directory.mkdir(parents=True, exist_ok=True)
            metrics = directory / "metrics.json"
            if not metrics.exists():
                while True:
                    state = gpu_state()
                    if args.device == "cpu":
                        selected = next(g for g in state if g["index"] == args.cpu_pinned_gpu)
                        break
                    selected = select_gpu(state, args.gpu_indices, args.share_gpu, args.min_free_gib)
                    if selected:
                        break
                    write_json(
                        output / "status.json",
                        {
                            "state": "WAITING_FOR_GPU_MEMORY" if args.share_gpu else "WAITING_FOR_IDLE_GPU",
                            "updated_at": now(),
                            "next": f"{entry['split']}/{entry['object']}",
                            "gpus": state,
                            "completed_objects": len(records),
                            "total_objects": len(datasets),
                        },
                    )
                    reason = "GPU memory headroom" if args.share_gpu else "an idle GPU"
                    print(f"[{now()}] Waiting for {reason}; existing tasks left running.", flush=True)
                    time.sleep(args.poll_seconds)
                env["CUDA_VISIBLE_DEVICES"] = selected["uuid"]
                command = eval_command(entry, checkpoint, args.seed, device, directory, args.threads)
                write_json(directory / "command.json", {"command": command, "gpu": selected, "started_at": now()})
                print(
                    f"[{now()}] Evaluating {entry['split']}/{entry['object']}: {entry['num_clips']} clips", flush=True
                )
                with (directory / "eval.log").open("w") as log:
                    process = subprocess.Popen(
                        command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                    )
                    write_json(
                        output / "status.json",
                        {
                            "state": "RUNNING",
                            "updated_at": now(),
                            "current": f"{entry['split']}/{entry['object']}",
                            "child_pid": process.pid,
                            "device": device,
                            "physical_gpu": selected["index"],
                            "shared_gpu": args.share_gpu,
                            "gpu_before_start": selected,
                            "completed_objects": len(records),
                            "total_objects": len(datasets),
                        },
                    )
                    try:
                        status = process.wait()
                    except BaseException:
                        os.killpg(process.pid, signal.SIGINT)
                        process.wait(timeout=30)
                        raise
                if status:
                    raise RuntimeError(f"Evaluation exited with {status}; see {directory / 'eval.log'}")
            records[(entry["split"], entry["object"])] = load_result(metrics, entry, checkpoint, args.seed, device)
            summarize(output, plan, records)
        write_json(
            output / "status.json",
            {
                "state": "COMPLETED",
                "updated_at": now(),
                "completed_objects": len(records),
                "total_objects": len(datasets),
            },
        )
        print(f"[{now()}] Complete: {output / 'README.md'}", flush=True)
    except BaseException as error:
        write_json(
            output / "status.json",
            {
                "state": "FAILED",
                "updated_at": now(),
                "error": str(error),
                "completed_objects": len(records),
                "total_objects": len(datasets),
            },
        )
        raise


if __name__ == "__main__":
    main()
