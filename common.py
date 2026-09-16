"""Shared file formats, model revisions, and atomic output helpers."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np

MODEL_REVISIONS = {
    "Alibaba-NLP/gte-base-en-v1.5": "a829fd0e060bb84554da0dfd354d0de0f7712b7f",
    "FacebookAI/roberta-base": "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "mistralai/Mistral-7B-Instruct-v0.3": "c170c708c41dac9275d15a8fff4eca08d52bab71",
}
TARGETS = {"qwen": "Qwen/Qwen2.5-7B-Instruct", "mistral": "mistralai/Mistral-7B-Instruct-v0.3"}
METHODS = ("topk", "label_mass", "fpa")


def model_kwargs(name, revision=None):
    if Path(name).expanduser().is_dir():
        return {}
    return {"revision": revision or MODEL_REVISIONS.get(name, "main")}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_arrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def check_output(directory, names, overwrite=False):
    directory = Path(directory)
    if not overwrite and any((directory / name).exists() for name in names):
        raise FileExistsError(f"Outputs already exist at {directory}. Choose another run directory or pass --overwrite.")
    directory.mkdir(parents=True, exist_ok=True)


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    os.close(fd)
    temporary = Path(filename)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def save_arrays(path, **arrays):
    with atomic_path(path) as temporary, temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def load_data(directory, task, limit_train=None, limit_test=None):
    root = Path(directory) / task
    metadata = read_json(root / "metadata.json")
    if metadata["task"] != task:
        raise ValueError("Task does not match data metadata")
    rows = []
    for split, limit in (("train", limit_train), (metadata["evaluation_split"], limit_test)):
        descriptor = metadata["files"][split]
        path = root / descriptor["file"]
        if sha256(path) != descriptor["sha256"]:
            raise ValueError(f"Data changed after preparation: {path}")
        with path.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if len(records) != descriptor["rows"] or not records:
            raise ValueError(f"Unexpected row count: {path}")
        ids = [row["id"] for row in records]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate IDs: {path}")
        if any(not 0 <= row["label"] < len(metadata["label_names"]) for row in records):
            raise ValueError(f"Evaluation and memory must have valid labels: {path}")
        rows.append(records[:limit])
    return *rows, metadata


def row_arrays(train, test):
    return dict(candidate_ids=np.asarray([r["id"] for r in train]),
                test_ids=np.asarray([r["id"] for r in test]),
                candidate_labels=np.asarray([r["label"] for r in train], dtype=np.int64),
                test_labels=np.asarray([r["label"] for r in test], dtype=np.int64))


def run_metadata(args, data_metadata, **extra):
    return dict(task=args.task, settings={k: str(v) if isinstance(v, Path) else v
                                       for k, v in vars(args).items()},
                source_data_metadata=data_metadata, numpy_version=np.__version__, **extra)


def aligned(left, right, keys):
    for key in keys:
        if not np.array_equal(left[key], right[key]):
            raise ValueError(f"Input artifacts disagree on {key}")
