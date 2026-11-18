"""Run model API on Laptop 1: python scripts/serve.py --host 0.0.0.0."""

import argparse, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    import uvicorn

    uvicorn.run("pathsnap.api.main:app", host=a.host, port=a.port, log_level="info")
