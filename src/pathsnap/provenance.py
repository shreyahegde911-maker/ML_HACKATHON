"""Read-only snapshot of the frozen inference and scoring implementation."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess

from pathsnap.candidates.generator import CandidateConfig
from pathsnap.data.alignment import align_reference_route
from pathsnap.features.engineering import FEATURES

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot():
    files = sorted((ROOT / "src/pathsnap").rglob("*.py"))
    files += [ROOT / name for name in ["configs/decoder.json", "requirements.txt", "scripts/serve.py", "scripts/launch.py", "ui/index.html"]]
    components = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    signature = inspect.signature(align_reference_route)
    alignment = {name: parameter.default for name, parameter in signature.parameters.items()
                 if parameter.default is not inspect.Parameter.empty}
    configuration = {
        "features": FEATURES,
        "feature_preprocessing": "Frozen model artifact; source-order four-feature engineering; no fit at inference",
        "candidate": asdict(CandidateConfig()),
        "decoder": json.loads((ROOT / "configs/decoder.json").read_text()),
        "reference_alignment": alignment,
        "scoring_eligible_status": "aligned",
        "projection": "Per-trip UTM chosen from median GPS longitude/latitude; no fitted transform",
    }
    core = {"model_sha256": sha256(ROOT / "artifacts/final/selected.joblib"),
            "components": components, "configuration": configuration}
    fingerprint = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
    return {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "pipeline_sha256": fingerprint,
        "configuration_sha256": hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest(),
        **core,
    }


def freeze():
    import joblib
    bundle = joblib.load(ROOT / "artifacts/final/selected.joblib")
    model = bundle["model"]
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "selected_model_class": type(model).__name__,
            "selected_model_parameters": model.get_params(),
            "artifact_features": bundle.get("features"),
            "pip_freeze": subprocess.check_output([str(__import__('sys').executable), "-m", "pip", "freeze"], text=True).splitlines(),
            **snapshot()}


def assert_frozen(manifest):
    current = snapshot()
    if current["pipeline_sha256"] != manifest["pipeline_sha256"]:
        raise RuntimeError("Frozen inference/scoring files or selected model changed; evaluation stopped")
    return current
