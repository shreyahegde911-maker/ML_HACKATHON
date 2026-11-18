"""Launcher checks avoid modifying real environments or stopping existing apps."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import launch


class LauncherTests(unittest.TestCase):
    def invoke(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return launch.main(args)

    def test_ui_only_needs_no_environment_install(self):
        with patch.object(launch.subprocess, "run") as run:
            self.assertEqual(launch.environment_python(ui_only=True), sys.executable)
            run.assert_not_called()

    def test_check_starts_no_services(self):
        with patch.object(launch, "validate_files"), patch.object(
            launch, "environment_python", return_value=sys.executable
        ), patch.object(launch, "start_process") as start:
            self.assertEqual(self.invoke(["--check"]), 0)
            start.assert_not_called()

    def test_existing_services_are_not_owned_or_stopped(self):
        with patch.object(launch, "validate_files"), patch.object(
            launch, "environment_python", return_value=sys.executable
        ), patch.object(launch, "service_ready", return_value=True), patch.object(
            launch, "start_process"
        ) as start, patch.object(
            launch, "stop_process"
        ) as stop, patch.object(
            launch.time, "sleep", side_effect=KeyboardInterrupt
        ):
            self.assertEqual(self.invoke(["--no-browser"]), 0)
            start.assert_not_called()
            stop.assert_not_called()

    def test_conflicting_port_starts_nothing(self):
        with patch.object(launch, "validate_files"), patch.object(
            launch, "environment_python", return_value=sys.executable
        ), patch.object(launch, "service_ready", return_value=False), patch.object(
            launch, "port_in_use", return_value=True
        ), patch.object(
            launch, "start_process"
        ) as start:
            self.assertEqual(self.invoke(["--no-browser"]), 1)
            start.assert_not_called()

    def test_ctrl_c_cleans_up_only_new_processes(self):
        api, ui = MagicMock(), MagicMock()
        api.poll.return_value = ui.poll.return_value = None
        with patch.object(launch, "validate_files"), patch.object(
            launch, "environment_python", return_value=sys.executable
        ), patch.object(launch, "service_ready", return_value=False), patch.object(
            launch, "port_in_use", return_value=False
        ), patch.object(
            launch, "wait_ready"
        ), patch.object(
            launch, "start_process", side_effect=[api, ui]
        ), patch.object(
            launch, "stop_process"
        ) as stop, patch.object(
            launch.time, "sleep", side_effect=KeyboardInterrupt
        ):
            self.assertEqual(self.invoke(["--no-browser"]), 0)
            self.assertEqual([c.args[0] for c in stop.call_args_list], [ui, api])

    def test_failed_start_cleans_up_already_launched_child(self):
        child = MagicMock()
        with patch.object(launch, "validate_files"), patch.object(
            launch, "environment_python", return_value=sys.executable
        ), patch.object(launch, "service_ready", return_value=False), patch.object(
            launch, "port_in_use", return_value=False
        ), patch.object(
            launch, "start_process", return_value=child
        ), patch.object(
            launch, "wait_ready", side_effect=RuntimeError("not ready")
        ), patch.object(
            launch, "stop_process"
        ) as stop:
            self.assertEqual(self.invoke(["--no-browser"]), 1)
            stop.assert_called_once_with(child)

    def test_ui_only_validation_does_not_require_models_or_data(self):
        with tempfile.TemporaryDirectory(prefix="PathSnap space & test ") as temp:
            root = Path(temp)
            (root / "ui").mkdir()
            (root / "ui/index.html").write_text("UI")
            with patch.object(launch, "ROOT", root):
                launch.validate_files(api=False)
                with self.assertRaisesRegex(RuntimeError, "selected.joblib"):
                    launch.validate_files(api=True)

    def test_duplicate_ports_rejected_before_setup(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launch.main(["--api-port", "8000", "--ui-port", "8000"])

    def test_windows_child_gets_its_own_process_group(self):
        with patch.object(launch.os, "name", "nt"), patch.object(
            launch.subprocess, "CREATE_NEW_PROCESS_GROUP", 512, create=True
        ), patch.object(launch.subprocess, "Popen") as popen:
            launch.start_process(["python.exe", "server.py"])
            self.assertEqual(popen.call_args.kwargs["creationflags"], 512)
            self.assertNotIn("start_new_session", popen.call_args.kwargs)

    def test_windows_shutdown_signals_only_owned_group(self):
        child = MagicMock()
        child.poll.return_value = None
        with patch.object(launch.os, "name", "nt"), patch.object(
            launch.signal, "CTRL_BREAK_EVENT", 1, create=True
        ):
            launch.stop_process(child)
        child.send_signal.assert_called_once_with(1)
        child.wait.assert_called_once_with(timeout=5)


if __name__ == "__main__":
    unittest.main()
