"""Reproducible trip-level development split; never reads trajectory contents."""

from pathlib import Path
import numpy as np
from .readers import DEVELOPMENT_PARTITIONS


def create_split_manifest(data_root, *, seed=42, validation_count=16):
    root = Path(data_root)
    ids = {}
    for partition in DEVELOPMENT_PARTITIONS:
        if not (root / partition).is_dir():
            raise FileNotFoundError(root / partition)
        ids[partition] = sorted(
            p.name
            for p in (root / partition).iterdir()
            if p.is_dir() and p.name.isdigit()
        )
    if len(ids["train"]) != 80 or len(ids["student_test"]) != 10:
        raise ValueError(
            f"Expected 80 training and 10 student_test trips, got { {k: len(v) for k,v in ids.items()} }"
        )
    if set(ids["train"]) & set(ids["student_test"]):
        raise ValueError("Trip IDs overlap between supplied partitions")
    if not 0 < validation_count < 80:
        raise ValueError("validation_count must be between 1 and 79")
    shuffled = np.random.default_rng(seed).permutation(ids["train"]).tolist()
    return {
        "seed": seed,
        "algorithm": "sorted IDs; numpy default_rng(seed).permutation; first IDs validate",
        "train": sorted(shuffled[validation_count:]),
        "validation": sorted(shuffled[:validation_count]),
        "student_test": ids["student_test"],
        "source_partitions": {
            "train": "train",
            "validation": "train",
            "student_test": "student_test",
        },
        "evaluator_testing": "excluded: final evaluation only",
    }
