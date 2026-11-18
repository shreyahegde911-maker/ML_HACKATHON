"""Shared standard-library launcher for start.sh and start.bat."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.request import Request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def say(message):
    print(f"[PathSnap] {message}", flush=True)


def environment_python(ui_only=False, skip_install=False):
    """Keep installation inside the project venv; the UI needs only stdlib."""
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "Python 3.12 is required. Install it and rerun the launcher."
        )
    if ui_only:
        return sys.executable
    directory = ROOT / ".venv"
    python = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        say("Creating the project virtual environment…")
        subprocess.run([sys.executable, "-m", "venv", str(directory)], check=True)
    version = subprocess.check_output(
        [str(python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        text=True,
    ).strip()
    if version != "3.12":
        raise RuntimeError(
            "The existing .venv must use Python 3.12. Recreate it before starting."
        )
    requirements = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    probe = (
        "import importlib.metadata as m,json,sys\n"
        "missing=[]\n"
        "for requirement in json.loads(sys.argv[1]):\n"
        " name,expected=requirement.split('==',1)\n"
        " try: actual=m.version(name)\n"
        " except m.PackageNotFoundError: actual=None\n"
        " if actual!=expected: missing.append(requirement)\n"
        "print(json.dumps(missing))\n"
    )
    missing = json.loads(
        subprocess.check_output(
            [str(python), "-c", probe, json.dumps(requirements)], text=True
        )
    )
    if missing:
        if skip_install:
            raise RuntimeError(
                "Missing or mismatched dependencies: " + ", ".join(missing)
            )
        say("Installing pinned dependencies into .venv (first start needs internet)…")
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
            check=True,
        )
    return str(python)


def validate_files(api=True):
    if not (ROOT / "ui/index.html").is_file():
        raise RuntimeError(
            "ui/index.html is missing. Run this from a complete project copy."
        )
    if not api:
        return
    for relative in (
        "artifacts/final/selected.joblib",
        "configs/decoder.json",
        "scripts/serve.py",
    ):
        if not (ROOT / relative).is_file():
            raise RuntimeError(
                f"Missing {relative}. Pull the complete repository; no retraining is required."
            )
    data = Path(
        os.environ.get("PATHSNAP_EVALUATOR_ROOT", ROOT / "Student folder/student_test")
    )
    trips = (
        sorted(p for p in data.iterdir() if p.is_dir() and p.name.isdigit())
        if data.is_dir()
        else []
    )
    if len(trips) != 10 or any(len(p.name) != 8 for p in trips):
        raise RuntimeError(f"Expected ten eight-digit trip folders in {data}.")
    for trip in trips:
        for suffix in ("track", "nodes", "arcs"):
            if not (trip / f"{trip.name}.{suffix}").is_file():
                raise RuntimeError(f"Missing {trip.name}.{suffix} in {trip}.")


def service_ready(service, port):
    url = f"http://127.0.0.1:{port}" + ("/health" if service == "API" else "/")
    try:
        with urlopen(url, timeout=1) as response:
            body = response.read()
        if service == "API":
            data = json.loads(body)
            return (
                data.get("status") == "ok"
                and data.get("model_ready") is True
                and data.get("trip_count") == 10
            )
        return b"PathSnap" in body and b'id="trip"' in body and b'id="run"' in body
    except (OSError, ValueError, URLError):
        return False


def port_in_use(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def wait_ready(service, port, process, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{service} exited with code {process.returncode}; check its log above."
            )
        if service_ready(service, port):
            return
        time.sleep(0.25)
    raise RuntimeError(
        f"{service} was not ready after {timeout} seconds; check its log above."
    )


def start_process(command):
    options = {"cwd": str(ROOT), "env": {**os.environ, "PYTHONUNBUFFERED": "1"}}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    # Inherit the terminal so actual inference request logs remain visible.
    return subprocess.Popen(command, **options)


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        process.kill()
        process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--api-only", action="store_true", help="Laptop 1: model/API only"
    )
    modes.add_argument(
        "--ui-only", action="store_true", help="Laptop 2: UI only, no ML install"
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Start, verify a prediction, and stop launcher-owned services",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Fail instead of installing dependencies",
    )
    parser.add_argument(
        "--check", action="store_true", help="Check setup without starting services"
    )
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=8001)
    args = parser.parse_args(argv)
    if not all(1 <= p <= 65535 for p in (args.api_port, args.ui_port)):
        parser.error("Ports must be between 1 and 65535")
    if not (args.api_only or args.ui_only) and args.api_port == args.ui_port:
        parser.error("API and UI must use different ports")
    processes = []
    try:
        validate_files(api=not args.ui_only)
        python = environment_python(args.ui_only, args.skip_install)
        if args.check:
            say("Setup is ready. No services were started.")
            return 0
        services = []
        if not args.ui_only:
            services.append(
                (
                    "API",
                    args.api_port,
                    [
                        python,
                        str(ROOT / "scripts/serve.py"),
                        "--host",
                        "0.0.0.0",
                        "--port",
                        str(args.api_port),
                    ],
                )
            )
        if not args.api_only:
            services.append(
                (
                    "UI",
                    args.ui_port,
                    [
                        python,
                        str(ROOT / "scripts/serve_ui.py"),
                        "--host",
                        "0.0.0.0",
                        "--port",
                        str(args.ui_port),
                    ],
                )
            )
        existing = {name: service_ready(name, port) for name, port, _ in services}
        # Check all ports before starting either service; never stop another app.
        for name, port, _ in services:
            if not existing[name] and port_in_use(port):
                raise RuntimeError(
                    f"Port {port} is occupied by another service or an unready {name}. Close it or choose a different port."
                )
        for name, port, command in services:
            if existing[name]:
                say(f"{name} already running on port {port}; reusing it.")
            else:
                say(f"Starting {name} on port {port}…")
                process = start_process(command)
                processes.append(process)
                wait_ready(name, port, process)
        api = f"http://localhost:{args.api_port}"
        url = (
            api + "/docs"
            if args.api_only
            else f"http://localhost:{args.ui_port}/?" + urlencode({"api": api})
        )
        say(f"Ready: {url}")
        if args.smoke_test:
            if not args.ui_only:
                endpoint = f"http://127.0.0.1:{args.api_port}"
                with urlopen(endpoint + "/trips", timeout=10) as response:
                    trip_id = json.load(response)["trips"][0]
                request = Request(
                    endpoint + "/predict",
                    data=json.dumps({"trip_id": trip_id}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(request, timeout=120) as response:
                    prediction = json.load(response)
                if prediction.get("trip_id") != trip_id or len(
                    prediction.get("predictions", [])
                ) != prediction.get("observation_count"):
                    raise RuntimeError(
                        "Prediction smoke test returned inconsistent observations"
                    )
                say(
                    f"Prediction smoke test passed: {trip_id}, {prediction['observation_count']} observations."
                )
            say("Startup/shutdown smoke test passed.")
            return 0
        if args.ui_only:
            say(
                "On Laptop 2, set the API address to http://LAPTOP1_LAN_IP:8000 and click Connect."
            )
        say(
            "Keep this window open. Ctrl+C stops only services started by this launcher."
        )
        if not args.no_browser:
            try:
                webbrowser.open(url)
            except (webbrowser.Error, OSError):
                say("Could not open a browser automatically; open the Ready URL above.")
        while True:
            for process in processes:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"A service exited with code {process.returncode}; check its log above."
                    )
            time.sleep(0.5)
    except KeyboardInterrupt:
        say("Stopping launcher-owned services…")
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        say(f"ERROR: {exc}")
        return 1
    finally:
        for process in reversed(processes):
            stop_process(process)


if __name__ == "__main__":
    sys.exit(main())
