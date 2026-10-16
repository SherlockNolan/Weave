#!/usr/bin/env python3
"""Record single-episode pickup evaluations, sequentially on physical GPU 0."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GPU0 = "GPU-6de172b1-a4eb-dd56-d8ae-00be059f483f"
CLIPS = (
    ("tripod", "sub1_tripod_075_v00"),
    ("clothesstand", "sub16_clothesstand_012_v00"),
    ("clothesstand", "sub9_clothesstand_039_v00"),
    ("whitechair", "sub7_whitechair_072"),
)
FRAME_FIELDS = (
    "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
    "object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w", "contact_label",
)


def gpu_rows():
    result = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"
    ], text=True)
    return {tuple(x.strip() for x in row) for row in csv.reader(result.splitlines()) if row}


def check_gpu(share):
    gpu = subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"
    ], text=True).strip().split(",")
    if gpu[0].strip() != GPU0 or int(gpu[1]) < 8192:
        raise RuntimeError("GPU 0 identity changed or less than 8 GiB is free")
    rows = gpu_rows()
    local = [row for row in rows if row[0] == GPU0]
    if local and not share:
        raise RuntimeError("GPU 0 is busy; sharing requires --share-weave-gpu0")
    if share:
        job = (REPO / "logs/training/latest-dual").resolve()
        if (job / "status").read_text().strip() != "RUNNING":
            raise RuntimeError("No running owned Weave training job")
        # Confirm the active torchrun belongs to this job and that no extra
        # compute process has appeared on its GPU before starting a recorder.
        processes = subprocess.check_output(["ps", "-eo", "args"], text=True)
        if str(job / "workers") not in processes or len(local) != 2:
            raise RuntimeError("Unexpected GPU 0 workload; refusing shared recording")
    return rows


def extract_clip(obj, name, path):
    with np.load(REPO / f"data/pickups/{obj}_pickups.npz", allow_pickle=False) as archive:
        index = list(archive["motion_names"].astype(str)).index(name)
        starts = np.r_[0, np.cumsum(archive["motion_lengths"])]
        arrays = {}
        for key in archive.files:
            if key in FRAME_FIELDS:
                arrays[key] = archive[key][starts[index]:starts[index + 1]].copy()
            elif key in ("motion_names", "motion_lengths", "object_names"):
                arrays[key] = archive[key][index:index + 1].copy()
            else:
                arrays[key] = archive[key].copy()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        # Frame the full reference path, with a fixed camera so motion remains visible.
        points = np.concatenate((arrays["body_pos_w"][:, 0, :2], arrays["object_pos_w"][:, :2]))
        center = (points.min(0) + points.max(0)) / 2
        return [float(center[0]) + 3, float(center[1]) - 3, 2.0], [float(center[0]), float(center[1]), .85]


def probe(path):
    report = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,avg_frame_rate:format=duration,size", "-of", "json", str(path)
    ], text=True)
    return json.loads(report)


def render(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline = check_gpu(args.share_weave_gpu0)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU0, CUDA_DEVICE_ORDER="PCI_BUS_ID",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", HYDRA_FULL_ERROR="1")
    cpus = ",".join(map(str, sorted(os.sched_getaffinity(0))[-4:]))
    checkpoint = Path((REPO / "logs/training/latest-chain/pickup-checkpoint.txt").read_text().strip())
    clips = []
    for obj, name in CLIPS:
        sequence = name.split("_v")[0]
        directory = output / sequence
        directory.mkdir(parents=True, exist_ok=True)
        motion = output / "inputs" / f"{name}.npz"
        eye, lookat = extract_clip(obj, name, motion)
        metrics_path = directory / "metrics.json"
        if not (args.reuse_existing and metrics_path.exists() and (directory / "episode.mp4").exists()):
            check_gpu(args.share_weave_gpu0)
            command = ["nice", "-n", "19", "taskset", "-c", cpus, "python", "-u", "scripts/rsl_rl/eval.py",
                       "--task", "G1-Inspire-HOI-Pickup-v0", "--headless", "--config-dir", "./configs/track",
                       "--config-name", "eval_pickups", "--checkpoint", str(checkpoint), "--num_envs", "1",
                       "--device", "cuda:0", "--video", "--output_dir", str(directory),
                       f"env.commands.motion.motion_files=[{motion}]", "env.viewer.resolution=[960,540]",
                       "env.viewer.origin_type=env", f"env.viewer.eye={eye}", f"env.viewer.lookat={lookat}",
                       "--kit_args=--/renderer/multiGpu/enabled=false --/renderer/multiGpu/maxGpuCount=1 "
                       "--/plugins/carb.tasking.plugin/threadCount=4 --/plugins/omni.tbb.globalcontrol/maxThreadCount=4"]
            print(f"Recording {name}", flush=True)
            with (directory / "eval.log").open("w") as log:
                subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        metrics = json.loads(metrics_path.read_text())
        if metrics["num_clips"] != 1 or not metrics["video"]["terminal_pose_before_reset"]:
            raise RuntimeError(f"Incomplete first-episode recording for {name}")
        video = directory / "episode.mp4"
        info = probe(video)
        assert int(info["streams"][0]["nb_frames"]) == metrics["video"]["frames"]
        clips.append({"sequence": sequence, "motion_name": name, "video": str(video),
                      "metrics": str(metrics_path), "success": metrics["clips"][0]["success"],
                      "cause": metrics["clips"][0]["cause"], "progress": metrics["clips"][0]["progress"],
                      "media": info})
        # Observe other tasks; never change their processes or driver settings.
        current = gpu_rows()
        other = {row for row in baseline if row[0] != GPU0}
        if not {(r[0], r[1]) for r in other}.issubset({(r[0], r[1]) for r in current}):
            raise RuntimeError("An existing GPU task disappeared; stopping the recording batch")
        print(f"Recorded {name}: success={clips[-1]['success']}, progress={clips[-1]['progress']:.3f}", flush=True)
    (output / "manifest.json").write_text(json.dumps({"checkpoint": str(checkpoint), "clips": clips}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "outputs/pickup-videos")
    parser.add_argument("--share-weave-gpu0", action="store_true", help="Allow sharing GPU 0 with the owned two-rank Weave job")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    os.chdir(REPO)
    with (REPO / "logs/preparation/pickup-video.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.share_weave_gpu0:
            render(args)
        else:
            with (REPO / "logs/preparation/gpu0.lock").open("w") as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                render(args)


if __name__ == "__main__":
    main()
