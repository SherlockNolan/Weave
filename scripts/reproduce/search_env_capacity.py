#!/usr/bin/env python3
"""Measure the dual-GPU training capacity using short, otherwise unchanged runs."""

import argparse
import json
import os
import re
import signal
import subprocess
import time
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def gpu_memory():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return {
        int(index): {"used_mib": int(used), "free_mib": int(free)}
        for index, used, free in (line.split(",") for line in output.splitlines())
    }


def descendants(parent):
    parents = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                status = entry.joinpath("status").read_text()
                parents[int(entry.name)] = int(re.search(r"^PPid:\s+(\d+)", status, re.M)[1])
            except (OSError, TypeError):
                continue
    found = {parent}
    while True:
        children = {pid for pid, ppid in parents.items() if ppid in found}
        if children <= found:
            return found
        found |= children


def stop_trial(process):
    # Only this isolated trial and its descendants; never signal unrelated GPU users.
    owned = descendants(process.pid)
    for pid in owned - {process.pid}:
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        for pid in owned:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        process.wait(timeout=10)


def save_report(output, report):
    temporary = output / "status.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output / "status.json")
    lines = [
        "# 双卡环境数量 smoke 搜索", "",
        f"更新时间：{now()}。状态：`{report['status']}`。", "",
        "完整九物体 D_train；从头训练；Isaac Sim 5.1；每档 5 次 PPO 更新；",
        "rollout 32、epochs 5、mini-batches 4、原网络和 torch.compile 均保持。",
        "共用 GPU 已获用户明确授权，既有任务保留。", "",
        "| 每卡环境 | 完整更新 | 最小空闲 MiB（0 / 1） | 结果 |",
        "| ---: | ---: | --- | --- |",
    ]
    labels = {"PASS": "通过", "PASS_LOW_MARGIN": "通过，但显存余量不足", "RUNNING": "运行中"}
    for trial in sorted(report["trials"], key=lambda trial: trial["envs_per_gpu"]):
        free = trial.get("min_free_mib", {})
        lines.append(
            f"| {trial['envs_per_gpu']} | {trial.get('completed_updates', 0)} | "
            f"{free.get('0', free.get(0, '-'))} / {free.get('1', free.get(1, '-'))} | "
            f"{labels.get(trial['result'], trial['result'])} |"
        )
    lines += ["", "当前 16,384/卡已有首次 PPO 更新 CUDA OOM 的记录，作为搜索上界。",
              "搜索粒度为 2,048 环境/卡；推荐档位要求两卡全程至少剩余 3,072 MiB。",
              "显存指标为每 2 秒采样的整卡占用，包含已有任务；可能遗漏更短的瞬时峰值。",
              "有限次更新只能验证短时容量，不能保证长期运行或复现论文指标。"]
    if report.get("recommended_envs_per_gpu"):
        count = report["recommended_envs_per_gpu"]
        lines += ["", f"推荐：每卡 **{count:,}**，双卡总计 **{2 * count:,}**。"]
    (output / "README.md").write_text("\n".join(lines) + "\n")


def run_trial(root, output, report, count):
    trial_dir = output / f"envs_{count}"
    trial_dir.mkdir()
    pointer = trial_dir / "job.path"
    env = os.environ.copy()
    env.update({
        "WEAVE_GPU_INDICES": "0,1", "WEAVE_ALLOW_GPU_SHARING": "1",
        "WEAVE_MIN_FREE_GPU_MIB": "35000",
        "WEAVE_TRAIN_ACTIVATION": "outputs/training-starts/2026-10-19/activate_sim51_driver.sh",
        "WEAVE_TRAIN_TASK": "G1-Inspire-HOI-v0", "WEAVE_TRAIN_CONFIG": "train_reference_dual",
        "WEAVE_TRAIN_ENVS_PER_GPU": str(count), "WEAVE_TRAIN_ITERATIONS": "5",
        "WEAVE_TRAIN_RUN_NAME": f"capacity-smoke-dual-{count}-sim51-shared",
        "WEAVE_JOB_PTR": str(pointer),
    })
    trial = {
        "envs_per_gpu": count, "started_at": now(), "result": "RUNNING",
        "completed_updates": 0, "peak_used_mib": {0: 0, 1: 0},
        "min_free_mib": {0: 10**9, 1: 10**9},
    }
    report["trials"].append(trial)
    report["current_envs_per_gpu"] = count
    save_report(output, report)
    with (trial_dir / "launcher.log").open("w") as launcher_log:
        process = subprocess.Popen(
            ["bash", "scripts/reproduce/train_gpu01.sh"], cwd=root, env=env,
            stdout=launcher_log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        trial["launcher_pid"] = process.pid
        started = time.monotonic()
        aborted = None
        with (trial_dir / "memory.jsonl").open("w") as memory_log:
            while True:
                memory = gpu_memory()
                for index in (0, 1):
                    trial["peak_used_mib"][index] = max(trial["peak_used_mib"][index], memory[index]["used_mib"])
                    trial["min_free_mib"][index] = min(trial["min_free_mib"][index], memory[index]["free_mib"])
                memory_log.write(json.dumps({"time": now(), "gpus": memory}) + "\n")
                memory_log.flush()
                if pointer.is_file():
                    job = Path(pointer.read_text().strip())
                    trial["job_dir"] = str(job)
                    log_path = job / "train.log"
                    if log_path.is_file():
                        text = log_path.read_text(errors="replace")
                        trial["completed_updates"] = len(set(re.findall(r"Learning iteration (\d+)/", text)))
                available_kib = next(
                    int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:")
                )
                if process.poll() is not None:
                    break
                if available_kib < 12 * 1024**2:
                    aborted = "ABORTED_LOW_SYSTEM_RAM"
                elif time.monotonic() - started > 1800:
                    aborted = "ABORTED_TIMEOUT"
                if aborted:
                    stop_trial(process)
                    break
                save_report(output, report)
                time.sleep(2)
    log_path = Path(trial.get("job_dir", trial_dir)) / "train.log"
    text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    trial["completed_updates"] = len(set(re.findall(r"Learning iteration (\d+)/", text)))
    trial["returncode"] = process.returncode
    trial["finished_at"] = now()
    if aborted:
        trial["result"] = aborted
    elif re.search(r"torch\.(?:cuda\.)?OutOfMemoryError:", text):
        trial["result"] = "CUDA_OOM"
    elif process.returncode != 0 or trial["completed_updates"] != 5:
        trial["result"] = "FAILED_OTHER"
    else:
        job = Path(trial["job_dir"])
        run = Path(json.loads((job / "launch.json").read_text())["checkpoint_dir"])
        recipe = json.loads((run / "training_recipe.json").read_text())
        if recipe["envs_per_rank"] != count or recipe["world_size"] != 2 or not (run / "model_4.pt").is_file():
            trial["result"] = "FAILED_VERIFICATION"
        else:
            trial["checkpoint"] = str(run / "model_4.pt")
            trial["result"] = "PASS" if min(trial["min_free_mib"].values()) >= 3072 else "PASS_LOW_MARGIN"
    (trial_dir / "result.json").write_text(json.dumps(trial, indent=2) + "\n")
    save_report(output, report)
    return trial["result"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "RUNNING", "started_at": now(), "trials": [],
        "world_size": 2, "iterations_per_trial": 5, "step_size": 2048,
        "minimum_free_gpu_mib": 3072, "known_oom_envs_per_gpu": 16384,
        "known_oom_job": str(root / "logs/training/2026-10-19_18-41-36_reference-scaled-dual-16384-sim51-shared"),
        "recommended_envs_per_gpu": None,
    }
    save_report(output, report)
    low, high = 0, 8  # Units of 2048; 16384 is an observed failure, not retried.
    try:
        while high - low > 1:
            middle = (low + high) // 2
            result = run_trial(root, output, report, middle * 2048)
            if result == "PASS":
                low = middle
                report["recommended_envs_per_gpu"] = low * 2048
            elif result in ("CUDA_OOM", "PASS_LOW_MARGIN"):
                high = middle
            else:
                report["status"] = "STOPPED_OTHER_FAILURE"
                break
            save_report(output, report)
        else:
            report["status"] = "COMPLETE" if low else "NO_PASSING_CONFIGURATION"
    except BaseException as error:
        report["status"] = "SEARCH_ERROR"
        report["error"] = repr(error)
        raise
    finally:
        report["finished_at"] = now()
        report["gpus_after"] = gpu_memory()
        save_report(output, report)


if __name__ == "__main__":
    main()
