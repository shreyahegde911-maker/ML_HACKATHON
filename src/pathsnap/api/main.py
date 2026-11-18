"""Laptop 1: FastAPI model/decoder service and server-only evaluation."""

import json, logging, os, time
from pathlib import Path
from functools import lru_cache
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .service import ROOT, predict, payload, reference_for_evaluation
from pathsnap.data.readers import load_trip_directory
from pathsnap.decoding import sequence_metrics
from pathsnap.runtime import data_root, evaluator_root, dataset_id, trip_ids
from pathsnap.provenance import snapshot

app = FastAPI(title="PathSnap", version="1.0.0")
app.mount("/vendor", StaticFiles(directory=ROOT / "ui/vendor"), name="ui-vendor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
log = logging.getLogger("uvicorn.error")


def root():
    return data_root()


def mode():
    return (
        "reserved-evaluator"
        if evaluator_root()
        else "student-development-demo"
    )


def trips():
    return trip_ids(root())


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
    trip_id: str = Field(pattern=r"^[0-9]{1,32}$")


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
        "status": "ok" if ready and count > 0 else "not_ready",
        "model_ready": ready,
        "trip_count": count,
        "mode": mode(),
        "dataset_id": dataset_id(),
    }


@app.get("/trips")
def catalog():
    ids = trips()
    return {"mode": mode(), "trips": ids, "trip_count": len(ids)}


@app.get("/system")
def system():
    state = snapshot()
    return {key: state[key] for key in ["git_commit", "pipeline_sha256", "model_sha256", "configuration_sha256", "configuration"]}


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
    log.info("trip=%s stage=prediction_complete; starting server-only reference evaluation", request.trip_id)
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
        if labels is None:
            evaluation_warning = "Accuracy unavailable because no reference was supplied."
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
