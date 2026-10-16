#!/usr/bin/env python3
"""Render all 12 pickup variants with one explicitly selected policy checkpoint."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from record_pickups import REPO, extract_clip, gpu_rows, probe

GROUPS = {
    "sub1_tripod_075": ("tripod", "三脚架地面拾取", "Tripod - floor pickup"),
    "sub16_clothesstand_012": ("clothesstand", "衣架地面拾取", "Clothes stand - floor pickup"),
    "sub9_clothesstand_039": ("clothesstand", "衣架放倒后再拾取", "Clothes stand - put down and pick up"),
    "sub7_whitechair_072": ("whitechair", "白椅子从承托面取下", "White chair - remove from fitted support"),
}


def ffmpeg(arguments):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *arguments], check=True)


def save_manifest(root, checkpoint, clips):
    report = {"checkpoint": str(checkpoint), "policy": "unified", "checkpoint_iteration": int(checkpoint.stem[6:]),
              "seed": 42, "task": "G1-Inspire-HOI-Pickup-v0",
              "scene_note": "Same pickup scene as specialist videos; whitechair has a fitted support absent from unified training.",
              "render_physics_pose_guard": True, "clips": sorted(clips, key=lambda c: c["motion_name"])}
    temp = root / "manifest.json.tmp"
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temp.replace(root / "manifest.json")
    return report


def record_variant(root, checkpoint, sequence, obj, name, gpu, uuid, cpus, reuse):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    motion = root / "inputs" / (name + ".npz")
    eye, lookat = extract_clip(obj, name, motion)
    # Leave room around both the feet and long objects throughout the reference.
    eye = [lookat[0] + 4.5, lookat[1] - 4.5, 3.0]
    metrics_path = directory / "metrics.json"
    reusable = False
    if reuse and metrics_path.is_file() and (directory / "episode.mp4").is_file():
        old = json.loads(metrics_path.read_text())
        reusable = Path(old["checkpoint"]).resolve() == checkpoint and old.get("video", {}).get("terminal_pose_before_reset")
    if not reusable:
        current_uuid = subprocess.check_output([
            "nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
        occupied = [r for r in gpu_rows() if r[0] == uuid]
        if current_uuid != uuid or occupied:
            raise RuntimeError(f"GPU {gpu} identity changed or is now busy; existing processes left running")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", HYDRA_FULL_ERROR="1")
        command = ["nice", "-n", "19", "taskset", "-c", cpus, sys.executable, "-u", "scripts/rsl_rl/eval.py",
                   "--task", "G1-Inspire-HOI-Pickup-v0", "--headless", "--config-dir", "./configs/track",
                   "--config-name", "eval_pickups", "--checkpoint", str(checkpoint), "--seed", "42",
                   "--num_envs", "1", "--device", "cuda:0", "--video", "--output_dir", str(directory),
                   f"env.commands.motion.motion_files=[{motion}]", "env.viewer.resolution=[960,540]",
                   "env.viewer.origin_type=env", f"env.viewer.eye={eye}", f"env.viewer.lookat={lookat}",
                   f"--kit_args=--/renderer/activeGpu={gpu} --/renderer/multiGpu/enabled=false --/renderer/multiGpu/maxGpuCount=1 "
                   "--/plugins/carb.tasking.plugin/threadCount=4 --/plugins/omni.tbb.globalcontrol/maxThreadCount=4"]
        print(f"GPU {gpu}: recording {name}", flush=True)
        with (directory / "eval.log").open("w") as log:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                # Kill only the recorder child created here, never another GPU workload.
                process.kill()
                process.wait()
                raise
            if code:
                raise subprocess.CalledProcessError(code, command)
    metrics = json.loads(metrics_path.read_text())
    if Path(metrics["checkpoint"]).resolve() != checkpoint or metrics["num_clips"] != 1:
        raise RuntimeError(f"Wrong checkpoint or clip count in {name}")
    if not metrics["video"]["terminal_pose_before_reset"]:
        raise RuntimeError(f"Recording did not reach an episode termination: {name}")
    item = metrics["clips"][0]
    if item["name"] != name:
        raise RuntimeError(f"Wrong reference in {name}")
    video = directory / "episode.mp4"
    media = probe(video)
    if int(media["streams"][0]["nb_frames"]) != metrics["video"]["frames"]:
        raise RuntimeError(f"Incomplete video in {name}")
    print(f"GPU {gpu}: {name}: success={item['success']}, progress={item['progress']:.3f}, cause={item['cause']}", flush=True)
    return {"sequence": sequence, "motion_name": name, "gpu": gpu, "video": str(video),
            "metrics": str(metrics_path), "success": item["success"], "cause": item["cause"],
            "progress": item["progress"], "steps": item["steps"], "reference_frames": item["length"], "media": media}


def compose(root, report):
    checkpoint_name = Path(report["checkpoint"]).stem
    for clip in report["clips"]:
        outcome = "SUCCESS" if clip["success"] else "FAIL - " + clip["cause"]
        title = GROUPS[clip["sequence"]][2] + " | " + clip["motion_name"]
        subtitle = f"Unified {checkpoint_name} | {outcome} | progress {100 * clip['progress']:.0f}% | seed 42"
        color = "#a4f7be" if clip["success"] else "#ffc4b8"
        filters = ("pad=iw:ih+80:0:80:color=0x17212b,"
                   f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{title}':"
                   "expansion=none:fontcolor=white:fontsize=20:x=16:y=12,"
                   f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{subtitle}':"
                   f"expansion=none:fontcolor={color}:fontsize=19:x=16:y=45,tpad=stop_mode=clone:stop_duration=1")
        target = root / (clip["motion_name"] + ".mp4")
        ffmpeg(["-threads", "2", "-i", clip["video"], "-vf", filters, "-an", "-c:v", "libx264", "-threads", "2",
                "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)])
        clip["labeled_video"] = str(target)
    groups = []
    for sequence, (_, chinese, _) in GROUPS.items():
        clips = sorted([c for c in report["clips"] if c["sequence"] == sequence], key=lambda c: c["motion_name"])
        # A sequence without variants has the same name as its sole labeled MP4.
        target = root / (sequence + ".mp4")
        if len(clips) > 1:
            listing = root / (sequence + ".concat")
            listing.write_text("".join(f"file '{c['labeled_video']}'\n" for c in clips))
            ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(target)])
            listing.unlink()
        groups.append({"sequence": sequence, "name": chinese, "successes": sum(c["success"] for c in clips),
                       "clips": len(clips), "video": str(target), "media": probe(target)})
    duration = max(float(g["media"]["format"]["duration"]) for g in groups)
    inputs, filters = [], []
    for i, group in enumerate(groups):
        inputs += ["-threads", "2", "-i", group["video"]]
        filters.append(f"[{i}:v]scale=768:496,setsar=1,tpad=stop_mode=clone:stop_duration={duration:.3f}[v{i}]")
    filters.append("[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[out]")
    overview = root / "overview.mp4"
    ffmpeg([*inputs, "-filter_complex_threads", "2", "-filter_complex", ";".join(filters), "-map", "[out]",
            "-t", str(duration), "-an", "-c:v", "libx264", "-threads", "2", "-crf", "21", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(overview)])
    ffmpeg(["-i", str(overview), "-ss", "1", "-frames:v", "1", str(root / "overview.jpg")])
    report.update(groups=groups, overview=str(overview))
    (root / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["# 统一策略执行四个拾取任务", "", f"- 权重：`{report['checkpoint']}`。这是训练到第 {report['checkpoint_iteration']:,} 次迭代的中间模型，目标为 100,000 次。",
             "- [四宫格总览](overview.mp4)：每个面板依次展示该任务的全部变体，短面板结束后保持最后一帧。",
             "- 每个变体固定 seed 42，从参考首帧开始，按原失败条件和参考末尾条件结束；每段末尾停留一秒。",
             "- 成功数仅统计本次逐条录像的单次执行，不是多种子成功率测试。GPU 仿真重跑可能有差异。",
             "- 场景沿用专项录像：白椅子使用拟合承托面。统一策略原始训练没有该承托面，这是额外场景条件。",
             "- `sub16_clothesstand_012` 来自 test split，其余三条来自 train split。",
             "- 每次图像采集均检查机器人和物体的 PhysX 位姿未被渲染改变。", "",
             "## 四个任务的全部变体", "", "| 任务 | 本次成功数 | 全变体视频 |", "| --- | ---: | --- |"]
    for group in groups:
        lines.append(f"| {group['name']} `{group['sequence']}` | {group['successes']}/{group['clips']} | [MP4]({Path(group['video']).name}) |")
    lines += ["", "## 逐变体结果", "", "| 变体 | 结果 | 动作进度 | 结束原因 | 视频 |", "| --- | --- | ---: | --- | --- |"]
    for clip in report["clips"]:
        lines.append(f"| `{clip['motion_name']}` | {'成功' if clip['success'] else '失败'} | {100*clip['progress']:.1f}% | `{clip['cause']}` | [MP4]({Path(clip['labeled_video']).name}) |")
    lines += ["", "各变体目录内 `metrics.json` 为本次带录像评估的结果，所有结果均保留，不按成败筛选案例。", ""]
    lines += ["## 重现录像", "", "在仓库根目录运行（脚本会拒绝占用已繁忙的 GPU）：", "", "```bash",
              "source scripts/reproduce/activate.sh", "python scripts/reproduce/render_policy_pickups.py \\",
              f"  --checkpoint '{report['checkpoint']}' \\",
              f"  --output '{root}' --gpus 6 7 --reuse-existing", "```", ""]
    (root / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[6])
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--compose-only", action="store_true")
    args = parser.parse_args()
    os.chdir(REPO)
    checkpoint, root = args.checkpoint.resolve(), args.output.resolve()
    if not checkpoint.is_file():
        parser.error("Checkpoint does not exist")
    root.mkdir(parents=True, exist_ok=True)
    if args.compose_only:
        compose(root, json.loads((root / "manifest.json").read_text()))
        return
    locks, uuids = [], {}
    for gpu in args.gpus:
        lock = (REPO / f"logs/preparation/gpu{gpu}.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
        data = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid,memory.used,utilization.gpu",
                                        "--format=csv,noheader,nounits"], text=True).strip().split(",")
        uuid = data[0].strip()
        if int(data[1]) > 64 or int(data[2]) or any(row[0] == uuid for row in gpu_rows()):
            raise RuntimeError(f"GPU {gpu} is busy; no existing workload will be stopped")
        uuids[gpu] = uuid
    baseline = sorted(gpu_rows())
    (root / "gpu-jobs-before.json").write_text(json.dumps(baseline, indent=2) + "\n")
    manifest = json.loads((REPO / "data/pickups/manifest.json").read_text())
    requests = [(c["sequence"], c["object"], c["motion_name"]) for c in manifest["clips"]]
    assert len(requests) == 12
    completed = []
    cpus = sorted(os.sched_getaffinity(0))

    def worker(index, gpu):
        affinity = ",".join(map(str, cpus[-4*(index+1):len(cpus)-4*index or None]))
        results = []
        for sequence, obj, name in requests[index::len(args.gpus)]:
            results.append(record_variant(root, checkpoint, sequence, obj, name, gpu, uuids[gpu], affinity, args.reuse_existing))
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(worker, i, gpu) for i, gpu in enumerate(args.gpus)]
        for future in concurrent.futures.as_completed(futures):
            completed.extend(future.result())
            save_manifest(root, checkpoint, completed)
    report = save_manifest(root, checkpoint, completed)
    report["checkpoint_sha256"] = hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest()
    compose(root, report)
    (root / "gpu-jobs-after.json").write_text(json.dumps(sorted(gpu_rows()), indent=2) + "\n")
    print("Completed all 12 variants: " + str(root / "README.md"), flush=True)


if __name__ == "__main__":
    main()
