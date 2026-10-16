#!/usr/bin/env python3
"""Label pickup episode videos and compose a four-panel overview."""
import argparse
import json
from pathlib import Path
import subprocess


def ffmpeg(arguments):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *arguments], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    report = json.loads((root / "manifest.json").read_text())
    labels = {"sub1_tripod_075": "Tripod - floor pickup",
              "sub16_clothesstand_012": "Clothes stand - floor pickup",
              "sub9_clothesstand_039": "Clothes stand - put down and pick up",
              "sub7_whitechair_072": "White chair - remove from fitted support"}
    durations = []
    for clip in report["clips"]:
        target = root / (clip["sequence"] + ".mp4")
        result = "COMPLETED" if clip["success"] else "STOPPED - " + clip["cause"]
        first = labels[clip["sequence"]] + " | " + clip["motion_name"]
        second = f"model_2999 | single attempt | {result} | progress {100 * clip['progress']:.0f}%"
        color = "#a4f7be" if clip["success"] else "#ffc4b8"
        filters = ("pad=iw:ih+80:0:80:color=0x17212b,"
                   f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{first}':"
                   "expansion=none:fontcolor=white:fontsize=21:x=16:y=12,"
                   f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{second}':"
                   f"expansion=none:fontcolor={color}:fontsize=19:x=16:y=45,"
                   "tpad=stop_mode=clone:stop_duration=1")
        ffmpeg(["-threads", "2", "-i", clip["video"], "-vf", filters, "-an", "-c:v", "libx264",
                "-threads", "2", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)])
        clip["labeled_video"] = str(target)
        durations.append(float(clip["media"]["format"]["duration"]) + 1)
    inputs, filters = [], []
    duration = max(durations)
    for index, clip in enumerate(report["clips"]):
        inputs += ["-threads", "2", "-i", clip["labeled_video"]]
        filters.append(f"[{index}:v]scale=768:496,setsar=1,tpad=stop_mode=clone:stop_duration={duration:.3f}[v{index}]")
    filters.append("[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[out]")
    overview = root / "overview.mp4"
    ffmpeg([*inputs, "-filter_complex_threads", "2", "-filter_complex", ";".join(filters),
            "-map", "[out]", "-t", str(duration), "-an", "-c:v", "libx264", "-threads", "2",
            "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(overview)])
    ffmpeg(["-i", str(overview), "-ss", "1.5", "-frames:v", "1", str(root / "overview.jpg")])
    report["overview"] = str(overview)
    (root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(overview)


if __name__ == "__main__":
    main()
