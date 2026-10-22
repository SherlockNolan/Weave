"""CPU-only, metadata-first catalog for recorded Inspire/Revo2 dataset replay.

Opening a catalog reads only small clip metadata. Loading one clip streams past
unselected compressed frames in bounded chunks and caches only its visual data;
no simulator, policy, GPU, or full-dataset allocation is involved.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from retarget_dataset_io import REPO, get_source_layout

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
VISUAL_FIELDS = ("joint_pos", "body_pos_w", "body_quat_w", "object_pos_w", "object_quat_w")
SOURCE_VISUAL_FIELDS = ("source_joint_pos", "source_body_pos_w", "source_body_quat_w")
MAX_STREAM_CHUNK = 4 * 1024 * 1024
CACHE_SCHEMA = "weave-recorded-dataset-viewer-v1"


@dataclass(frozen=True)
class DatasetEntry:
    source: str
    split: str
    object: str
    name: str
    index: int
    frame_start: int
    frames: int
    fps: float
    hand: str
    method: str
    preview_only: bool = False
    review_status: str = ""
    label_suffix: str = ""

    @property
    def key(self) -> str:
        """Include archive and index so repeated sequence names remain distinct."""
        return f"{self.hand}/{self.method}/{self.split}/{self.source}#{self.index}/{self.name}"

    @property
    def label(self) -> str:
        hand = "Inspire" if self.hand == "inspire" else f"Revo2/{self.method}"
        obj = OBJECT_LABELS.get(self.object, self.object)
        return f"{self.name} · {obj} · {self.split} · {hand}{self.label_suffix}"

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.fps


def _metadata(path: Path) -> tuple[list[str], list[str], np.ndarray, float]:
    """NpzFile reads only explicitly accessed ZIP members, not frame arrays."""
    with np.load(path, allow_pickle=False) as data:
        names = [str(value) for value in data["motion_names"]]
        objects = [str(value) for value in data["object_names"]]
        lengths = np.asarray(data["motion_lengths"], dtype=np.int64)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if len(names) != len(objects) or lengths.shape != (len(names),):
        raise ValueError(f"Clip metadata length mismatch: {path}")
    if np.any(lengths <= 0) or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid clip frame count or FPS: {path}")
    return names, objects, lengths, fps


def read_frame_range(stream, start: int, stop: int) -> np.ndarray:
    """Read one C-contiguous NPY frame slice with bounded decompression memory.

    Seeking a ZipExtFile can internally request the entire skipped byte count.
    Explicit small reads keep that cost bounded even for a late clip in a
    multi-gigabyte member. Only the selected output is materialized.
    """
    version = np.lib.format.read_magic(stream)
    readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
    if version not in readers:
        raise ValueError(f"Unsupported reference NPY header version: {version}")
    shape, fortran, dtype = readers[version](stream)
    if fortran or dtype.hasobject or not shape:
        raise ValueError("Reference frame arrays must be numeric C-contiguous arrays")
    if not 0 <= start < stop <= shape[0]:
        raise ValueError(f"Invalid frame interval [{start}, {stop}) for {shape}")
    stride = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
    skipped = start * stride
    while skipped:
        chunk = stream.read(min(skipped, MAX_STREAM_CHUNK))
        if not chunk:
            raise ValueError("Truncated NPY member while skipping unselected frames")
        skipped -= len(chunk)
    result = np.empty((stop - start, *shape[1:]), dtype=dtype)
    target = memoryview(result).cast("B")
    copied = 0
    while copied < len(target):
        chunk = stream.read(min(len(target) - copied, MAX_STREAM_CHUNK))
        if not chunk:
            raise ValueError("Truncated NPY member while reading selected frames")
        target[copied : copied + len(chunk)] = chunk
        copied += len(chunk)
    return result


class DatasetCatalog:
    """Actual train/test references plus only locally converted Revo2 clips."""

    def __init__(self, repo: Path = REPO, cache_dir: Path | None = None):
        self.repo = Path(repo).resolve()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.repo / "outputs/viser/dataset_replay"
        entries = []
        for split in ("train", "test"):
            for path in sorted((self.repo / "data" / split).glob("*.npz")):
                names, objects, lengths, fps = _metadata(path)
                start = 0
                for index, (name, obj, length) in enumerate(zip(names, objects, lengths, strict=True)):
                    entries.append(
                        DatasetEntry(
                            source=str(path.relative_to(self.repo)),
                            split=split,
                            object=obj,
                            name=name,
                            index=index,
                            frame_start=start,
                            frames=int(length),
                            fps=fps,
                            hand="inspire",
                            method="original",
                        )
                    )
                    start += int(length)
        # Converted directory names are methods, never evidence of train/test.
        # A real source_split in provenance is required to prevent held-out
        # motions (e.g. sub16_clothesstand_012) being mislabeled as training.
        for manifest_path in sorted((self.repo / "data/revo2").glob("*/manifest.json")):
            manifest = json.loads(manifest_path.read_text())
            for clip in manifest.get("clips", []):
                split = clip.get("source_split")
                if split not in ("train", "test"):
                    continue
                output = clip.get("output")
                if not output:
                    continue
                path = Path(output)
                if not path.is_absolute():
                    path = self.repo / path
                if not path.is_file():
                    continue
                names, objects, lengths, fps = _metadata(path)
                if len(names) != 1 or names[0] != clip["motion_name"]:
                    raise ValueError(f"Converted clip provenance disagrees with arrays: {path}")
                entries.append(
                    DatasetEntry(
                        source=str(path.resolve().relative_to(self.repo)),
                        split=split,
                        object=objects[0],
                        name=names[0],
                        index=0,
                        frame_start=0,
                        frames=int(lengths[0]),
                        fps=fps,
                        hand="revo2",
                        method=str(clip.get("method", manifest.get("method", manifest_path.parent.name))),
                        preview_only=not clip.get("train_ready", False),
                        review_status=str(clip.get("offline_gate_status", clip.get("status", ""))),
                    )
                )
        labels = Counter(entry.label for entry in entries)
        entries = [
            replace(entry, label_suffix=f" · {entry.source} #{entry.index}") if labels[entry.label] > 1 else entry
            for entry in entries
        ]
        self.entries = tuple(
            sorted(entries, key=lambda t: (t.hand, t.method, t.split, t.object, t.name, t.source, t.index))
        )
        self.tasks = self.entries
        self.by_key = {entry.key: entry for entry in self.entries}
        if len(self.by_key) != len(self.entries):
            raise ValueError("Duplicate dataset catalog entries")
        self.by_label = {entry.label: entry for entry in self.entries}
        if len(self.by_label) != len(self.entries):
            raise ValueError("Duplicate dataset labels after archive disambiguation")

    def search(self, query="", split="all", hand="all", method="all", object="all") -> list[DatasetEntry]:
        words = query.casefold().split()
        if hand == "both":
            hand = "revo2"  # Only converted clips contain both recorded hands.
        return [
            entry
            for entry in self.entries
            if (split == "all" or entry.split == split)
            and (hand == "all" or entry.hand == hand)
            and (method == "all" or entry.method == method)
            and (object == "all" or entry.object == object)
            and all(word in f"{entry.label} {entry.object} {entry.source}".casefold() for word in words)
        ]

    def load(self, entry: DatasetEntry | str) -> dict[str, np.ndarray]:
        """Return selected pose buffers, caching independently from raw data."""
        if isinstance(entry, str):
            entry = self.by_key[entry]
        if self.by_key.get(entry.key) != entry:
            raise ValueError("Entry does not belong to the current dataset catalog")
        source = self.repo / entry.source
        stat = source.stat()
        stamp = [CACHE_SCHEMA, entry.key, stat.st_size, stat.st_mtime_ns, entry.frame_start, entry.frames]
        digest = hashlib.sha256(json.dumps(stamp).encode()).hexdigest()
        path = self.cache_dir / "clips" / f"{digest}.npz"
        if path.is_file():
            try:
                with np.load(path, allow_pickle=False) as data:
                    arrays = {key: data[key] for key in data.files}
                self._validate(arrays, entry)
                return arrays
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                # This is a disposable selected-clip cache; raw source archives
                # remain read-only. Rebuild an interrupted/corrupt cache.
                path.unlink(missing_ok=True)
        fields = VISUAL_FIELDS + (SOURCE_VISUAL_FIELDS if entry.hand == "revo2" else ())
        arrays = {}
        with zipfile.ZipFile(source) as archive:
            for field in fields:
                with archive.open(f"{field}.npy") as stream:
                    arrays[field] = read_frame_range(stream, entry.frame_start, entry.frame_start + entry.frames)
        with np.load(source, allow_pickle=False) as data:
            if entry.hand == "inspire":
                layout = get_source_layout()
                arrays["joint_names"] = np.asarray(layout["joint_names"])
                arrays["body_names"] = np.asarray(layout["body_names"])
            else:
                for key in ("joint_names", "body_names", "source_joint_names", "source_body_names"):
                    arrays[key] = data[key]
        arrays.update(
            fps=np.asarray([entry.fps]),
            motion_names=np.asarray([entry.name]),
            motion_lengths=np.asarray([entry.frames], dtype=np.int64),
            object_names=np.asarray([entry.object]),
            source_split=np.asarray([entry.split]),
            preview_only=np.asarray([entry.preview_only]),
        )
        self._validate(arrays, entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, **arrays)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return arrays

    @staticmethod
    def _validate(data: dict[str, np.ndarray], entry: DatasetEntry) -> None:
        if (
            data["motion_names"].tolist() != [entry.name]
            or data["motion_lengths"].tolist() != [entry.frames]
            or data["object_names"].tolist() != [entry.object]
            or data["source_split"].tolist() != [entry.split]
            or data["fps"].tolist() != [entry.fps]
        ):
            raise ValueError(f"Cached clip metadata disagrees with source catalog: {entry.source}")
        prefixes = ("", "source_") if entry.hand == "revo2" else ("",)
        for prefix in prefixes:
            n_joints = len(data[prefix + "joint_names"])
            n_bodies = len(data[prefix + "body_names"])
            shapes = {
                prefix + "joint_pos": (entry.frames, n_joints),
                prefix + "body_pos_w": (entry.frames, n_bodies, 3),
                prefix + "body_quat_w": (entry.frames, n_bodies, 4),
            }
            for field, shape in shapes.items():
                if data[field].shape != shape or not np.isfinite(data[field]).all():
                    raise ValueError(f"Invalid {field} pose buffer in {entry.source}")
        for field, width in (("object_pos_w", 3), ("object_quat_w", 4)):
            if data[field].shape != (entry.frames, width) or not np.isfinite(data[field]).all():
                raise ValueError(f"Invalid {field} pose buffer in {entry.source}")
