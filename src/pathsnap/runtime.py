"""Portable local deployment settings; no model or reference access."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def local_settings():
    configured = os.environ.get("PATHSNAP_LOCAL_CONFIG")
    if configured == "":
        return {}
    path = Path(configured) if configured else ROOT / ".pathsnap-local.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def evaluator_root():
    value = os.environ.get("PATHSNAP_EVALUATOR_ROOT") or local_settings().get("evaluator_root")
    if not value:
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def data_root():
    return evaluator_root() or ROOT / "Student folder/student_test"


def dataset_id():
    return hashlib.sha256(str(data_root().resolve()).encode()).hexdigest()


def trip_ids(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.iterdir() if p.is_dir()
                  and p.name.isascii() and p.name.isdigit() and len(p.name) <= 32)
