"""Laptop 1: FastAPI model/decoder service and server-only evaluation."""

import json, logging, os, time
from pathlib import Path
from functools import lru_cache
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .service import ROOT, predict, payload, reference_for_evaluation
from pathsnap.data.readers import load_trip_directory
from pathsnap.decoding import sequence_metrics

app = FastAPI(title="PathSnap", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
log = logging.getLogger("uvicorn.error")


def root():
    return Path(
        os.environ.get("PATHSNAP_EVALUATOR_ROOT", ROOT / "Student folder/student_test")
    ).resolve()


def mode():
    return (
        "reserved-evaluator"
        if os.environ.get("PATHSNAP_EVALUATOR_ROOT")
        else "student-development-demo"
    )


def trips():
    return (
        sorted(
            p.name
            for p in root().iterdir()
            if p.is_dir() and len(p.name) == 8 and p.name.isascii() and p.name.isdigit()
        )
        if root().is_dir()
        else []
    )


@lru_cache(maxsize=1)
def _load_bundle(path, modified_ns, size):
    return joblib.load(path)


def bundle():
    path = ROOT / "artifacts/final/selected.joblib"
    stamp = path.stat()
    return _load_bundle(str(path), stamp.st_mtime_ns, stamp.st_size)


def settings():
    path = ROOT / "configs/decoder.json"
    return json.loads(path.read_text()) if path.exists() else {}


class Request(BaseModel):
    trip_id: str = Field(pattern=r"^\d{8}$")


@app.get("/health")
def health():
    ready = False
    try:
        loaded = bundle()
        ready = "model" in loaded and loaded.get("features") == [
            "distance_m",
            "heading_difference",
            "speed_mps",
            "time_gap_s",
        ]
    except Exception:
        log.warning("Model artifact is not ready", exc_info=True)
    count = len(trips())
    return {
        "status": "ok" if ready and count == 10 else "not_ready",
        "model_ready": ready,
        "trip_count": count,
        "mode": mode(),
    }


@app.get("/trips")
def catalog():
    return {"mode": mode(), "trips": trips(), "expected_count": 10}


@app.post("/predict")
def inference(request: Request):
    if request.trip_id not in trips():
        raise HTTPException(404, "Unknown configured trip")
    tick = time.perf_counter()
    directory = root() / request.trip_id
    try:
        trip = load_trip_directory(directory)
        pointwise, decoded, _ = predict(trip, bundle()["model"], settings())
    except (FileNotFoundError, ValueError) as exc:
        log.exception("Prediction failed for %s", request.trip_id)
        raise HTTPException(422, str(exc)) from exc
    # A damaged optional reference must not discard valid predictions. Keep the
    # private error in server logs, return no accuracy, and never expose labels.
    evaluation_warning = None
    try:
        labels = reference_for_evaluation(trip, directory)
        metrics = (
            {
                "classifier": sequence_metrics(trip, pointwise, labels),
                "hmm": sequence_metrics(trip, decoded["predictions"], labels),
            }
            if labels is not None
            else None
        )
    except (OSError, ValueError) as exc:
        log.warning("Reference evaluation unavailable for %s: %s", request.trip_id, exc)
        metrics = None
        evaluation_warning = (
            "Accuracy unavailable because the reference data could not be evaluated."
        )
    result = payload(
        trip, pointwise, decoded, metrics, time.perf_counter() - tick, mode()
    )
    if evaluation_warning:
        result["evaluation_warning"] = evaluation_warning
    log.info(
        "trip=%s observations=%d elapsed=%.3fs mode=%s metrics=%s",
        request.trip_id,
        len(trip["observations"]),
        result["inference_s"],
        mode(),
        metrics,
    )
    return result


@app.get("/")
def ui():
    return FileResponse(ROOT / "ui/index.html")
