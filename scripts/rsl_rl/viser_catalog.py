"""Index reference tasks on CPU and extract only the selected clip for evaluation."""

import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OBJECT_LABELS = {
    "clothesstand": "衣架",
    "floorlamp": "落地灯",
    "largebox": "大箱子",
    "largetable": "大桌子",
    "smallbox": "小箱子",
    "smalltable": "小桌子",
    "tripod": "三脚架",
    "whitechair": "白椅子",
    "woodchair": "木椅子",
}
FRAME_FIELDS = {
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "object_pos_w",
    "object_quat_w",
    "object_lin_vel_w",
    "object_ang_vel_w",
    "contact_label",
}


@dataclass(frozen=True)
class ReferenceTask:
    source: str
    split: str
    object: str
    name: str
    index: int
    frames: int
    pickup: bool = False

    @property
    def key(self):
        return f"{self.split}/{self.object}/{self.name}"

    @property
    def label(self):
        marker = "★ " if self.pickup else ""
        return f"{marker}{self.name} · {self.object} / {OBJECT_LABELS.get(self.object, self.object)} · {self.split}"

    @property
    def needs_support(self):
        return self.name == "sub7_whitechair_072"


class TaskCatalog:
    def __init__(self, scope="all", repo=REPO):
        self.repo = Path(repo)
        sources = sorted((self.repo / "data/train").glob("*.npz"))
        sources += sorted((self.repo / "data/test").glob("*.npz"))
        manifest = self.repo / "data/pickups/manifest.json"
        pickups = json.loads(manifest.read_text())["clips"] if manifest.is_file() else []
        focus = {(clip["source"], clip["motion_name"]) for clip in pickups}
        fingerprint = [(str(p.relative_to(self.repo)), p.stat().st_size, p.stat().st_mtime_ns) for p in sources]
        digest = hashlib.sha256(json.dumps([fingerprint, sorted(focus)]).encode()).hexdigest()
        cache = self.repo / "outputs/viser/catalog.json"
        saved = json.loads(cache.read_text()) if cache.is_file() else {}
        if saved.get("fingerprint") == digest:
            tasks = [ReferenceTask(**entry) for entry in saved["tasks"]]
        else:
            tasks = []
            for path in sources:
                relative = str(path.relative_to(self.repo))
                with np.load(path, allow_pickle=True) as data:
                    names = data["motion_names"].astype(str)
                    objects = data["object_names"].astype(str)
                    lengths = data["motion_lengths"]
                    for index, (name, obj, frames) in enumerate(zip(names, objects, lengths, strict=True)):
                        tasks.append(
                            ReferenceTask(
                                relative,
                                path.parent.name,
                                str(obj),
                                str(name),
                                index,
                                int(frames),
                                (relative, str(name)) in focus,
                            )
                        )
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"fingerprint": digest, "tasks": [asdict(t) for t in tasks]}))
        self.tasks = sorted(
            (t for t in tasks if scope == "all" or t.split == scope or (scope == "pickups" and t.pickup)),
            key=lambda t: (not t.pickup, t.object, t.name, t.split),
        )
        if not self.tasks:
            raise ValueError(f"No local reference tasks found for scope {scope!r}")
        self.by_key = {t.key: t for t in self.tasks}
        self.by_label = {t.label: t for t in self.tasks}

    def find(self, motion_files):
        """Prefer the explicitly supplied clip, including the existing pickup helper."""
        for path in motion_files:
            with np.load(path, allow_pickle=True) as archive:
                name = str(archive["motion_names"][0])
            candidates = [t for t in self.tasks if t.name == name]
            if candidates:
                return next((t for t in candidates if t.pickup), candidates[0])
        return self.tasks[0]

    def search(self, query="", split="all", obj="all"):
        words = query.casefold().split()
        return [
            t
            for t in self.tasks
            if (split == "all" or t.split == split or (split == "pickups" and t.pickup))
            and (obj == "all" or t.object == obj)
            and all(word in (t.label + (" pickup 拾取" if t.pickup else "")).casefold() for word in words)
        ]

    def extract(self, task):
        """Read selected frame ranges without materializing the full archive in RAM/GPU."""
        source = self.repo / task.source
        stat = source.stat()
        stamp = {"source": task.source, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "index": task.index}
        directory = self.repo / "outputs/viser/inputs" / task.split / task.object
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{task.name}.npz"
        metadata = path.with_suffix(".json")
        if not (path.is_file() and metadata.is_file() and json.loads(metadata.read_text()) == stamp):
            with np.load(source, allow_pickle=True) as data:
                starts = np.r_[0, np.cumsum(data["motion_lengths"], dtype=np.int64)]
                start, stop = int(starts[task.index]), int(starts[task.index + 1])
            arrays = {}
            with zipfile.ZipFile(source) as archive:
                for member in archive.namelist():
                    key = member.removesuffix(".npy")
                    if not member.endswith(".npy"):
                        continue
                    with archive.open(member) as stream:
                        if key in FRAME_FIELDS:
                            arrays[key] = self._read_frames(stream, start, stop)
                        else:
                            value = np.lib.format.read_array(stream, allow_pickle=True)
                            arrays[key] = (
                                value[task.index : task.index + 1]
                                if key in {"motion_names", "motion_lengths", "object_names"}
                                else value
                            )
            try:
                with tempfile.NamedTemporaryFile(dir=directory, suffix=".npz", delete=False) as stream:
                    temporary = Path(stream.name)
                    np.savez_compressed(stream, **arrays)
                os.replace(temporary, path)
                metadata.write_text(json.dumps(stamp))
            finally:
                if "temporary" in locals():
                    temporary.unlink(missing_ok=True)
        with np.load(path) as data:
            points = np.concatenate((data["body_pos_w"][:, 0, :2], data["object_pos_w"][:, :2]))
        center = (points.min(0) + points.max(0)) / 2
        lookat = (float(center[0]), float(center[1]), 0.85)
        eye = (lookat[0] + 4.5, lookat[1] - 4.5, 3.0)
        return path, eye, lookat

    @staticmethod
    def _read_frames(stream, start, stop):
        version = np.lib.format.read_magic(stream)
        readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
        if version not in readers:
            raise ValueError(f"Unsupported reference NPY header: {version}")
        shape, fortran, dtype = readers[version](stream)
        if fortran or dtype.hasobject:
            raise ValueError("Reference frame arrays must use numeric C-contiguous storage")
        stride = int(np.prod(shape[1:])) * dtype.itemsize
        stream.seek(start * stride, os.SEEK_CUR)
        raw = stream.read((stop - start) * stride)
        return np.frombuffer(raw, dtype=dtype).reshape((stop - start, *shape[1:])).copy()
