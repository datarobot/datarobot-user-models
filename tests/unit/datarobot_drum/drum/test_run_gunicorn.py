#
#  Copyright 2026 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import os
import signal
import subprocess
import sys
import textwrap
import time
from unittest.mock import MagicMock, patch

import pytest

from datarobot_drum.drum.gunicorn import run_gunicorn
from datarobot_drum.drum.gunicorn.run_gunicorn import (
    _FORWARDED_SIGNALS,
    _install_signal_forwarder,
    main_gunicorn,
)


class TestInstallSignalForwarder:
    def test_installs_handler_for_each_forwarded_signal(self):
        proc = MagicMock(pid=4242)

        with patch("datarobot_drum.drum.gunicorn.run_gunicorn.signal.signal") as sig_signal:
            _install_signal_forwarder(proc)

        installed = {call.args[0] for call in sig_signal.call_args_list}
        assert installed == set(_FORWARDED_SIGNALS)

    def test_handler_forwards_signal_to_child(self):
        proc = MagicMock(pid=4242)
        captured = {}

        def capture(sig, handler):
            captured[sig] = handler

        with patch("datarobot_drum.drum.gunicorn.run_gunicorn.signal.signal", side_effect=capture):
            _install_signal_forwarder(proc)

        captured[signal.SIGTERM](signal.SIGTERM, None)
        proc.send_signal.assert_called_once_with(signal.SIGTERM)

    def test_handler_ignores_already_exited_child(self):
        proc = MagicMock(pid=4242)
        proc.send_signal.side_effect = ProcessLookupError
        captured = {}

        with patch(
            "datarobot_drum.drum.gunicorn.run_gunicorn.signal.signal",
            side_effect=lambda s, h: captured.setdefault(s, h),
        ):
            _install_signal_forwarder(proc)

        captured[signal.SIGTERM](signal.SIGTERM, None)  # must not raise

    def test_skips_signals_that_are_unavailable_on_this_platform(self):
        proc = MagicMock(pid=4242)

        def fail_on_sigquit(sig, _handler):
            if sig == signal.SIGQUIT:
                raise OSError("not supported")

        with patch(
            "datarobot_drum.drum.gunicorn.run_gunicorn.signal.signal", side_effect=fail_on_sigquit
        ):
            _install_signal_forwarder(proc)  # must not raise


class TestMainGunicorn:
    """Verify main_gunicorn() wires up the forwarder before waiting on the child."""

    @pytest.fixture(autouse=True)
    def _stub_config_path(self, tmp_path, monkeypatch):
        # main_gunicorn() resolves the config path from __file__; redirect both the
        # parent lookup and the file-existence check at our temp directory.
        config = tmp_path / "gunicorn.conf.py"
        config.write_text("")
        # NB: `parent` is a reserved MagicMock constructor kwarg (it sets the
        # mock's parent, not a `.parent` attribute), so build the chain by
        # assigning `.resolve().parent` explicitly instead.
        path_instance = MagicMock()
        path_instance.resolve.return_value.parent = tmp_path
        monkeypatch.setattr(run_gunicorn, "Path", MagicMock(return_value=path_instance))
        return config

    def test_installer_runs_after_popen_and_before_wait(self, monkeypatch):
        events = []
        fake_proc = MagicMock(pid=9999)
        fake_proc.wait.side_effect = lambda: events.append("wait") or 0

        popen = MagicMock(side_effect=lambda *_a, **_kw: events.append("popen") or fake_proc)
        installer = MagicMock(side_effect=lambda _p: events.append("install"))

        monkeypatch.setattr(run_gunicorn, "subprocess", MagicMock(Popen=popen))
        monkeypatch.setattr(run_gunicorn, "_install_signal_forwarder", installer)

        main_gunicorn()

        assert events == ["popen", "install", "wait"]
        installer.assert_called_once_with(fake_proc)

    def test_exits_with_child_returncode_on_failure(self, monkeypatch):
        fake_proc = MagicMock(pid=1, wait=MagicMock(return_value=3))
        monkeypatch.setattr(
            run_gunicorn, "subprocess", MagicMock(Popen=MagicMock(return_value=fake_proc))
        )
        monkeypatch.setattr(run_gunicorn, "_install_signal_forwarder", MagicMock())

        with pytest.raises(SystemExit) as exc:
            main_gunicorn()
        assert exc.value.code == 3


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics required")
class TestSigtermPropagationEndToEnd:
    """Black-box: run a parent shaped like the new main_gunicorn() and confirm
    SIGTERM to the parent reaches the child."""

    def test_sigterm_reaches_child(self, tmp_path):
        marker = tmp_path / "marker.txt"

        child_src = textwrap.dedent(
            """
            import signal, sys, time
            marker = sys.argv[1]

            def on_term(sig, frame):
                with open(marker, 'a') as f:
                    f.write('CHILD-GOT-SIGTERM\\n')
                sys.exit(0)

            signal.signal(signal.SIGTERM, on_term)
            with open(marker, 'a') as f:
                f.write('CHILD-READY\\n')
            while True:
                time.sleep(0.1)
            """
        )
        parent_src = textwrap.dedent(
            """
            import subprocess, sys, signal

            proc = subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])

            def forward(sig, _frame):
                try:
                    proc.send_signal(sig)
                except ProcessLookupError:
                    pass

            for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
                signal.signal(s, forward)

            sys.exit(proc.wait())
            """
        )
        child_py = tmp_path / "child.py"
        child_py.write_text(child_src)
        parent_py = tmp_path / "parent.py"
        parent_py.write_text(parent_src)

        parent = subprocess.Popen([sys.executable, str(parent_py), str(child_py), str(marker)])
        try:
            for _ in range(50):
                if marker.exists() and "CHILD-READY" in marker.read_text():
                    break
                time.sleep(0.1)
            else:
                pytest.fail("child never became ready")

            os.kill(parent.pid, signal.SIGTERM)
            parent.wait(timeout=10)
        finally:
            if parent.poll() is None:
                parent.kill()

        contents = marker.read_text() if marker.exists() else ""
        assert "CHILD-GOT-SIGTERM" in contents
        assert parent.returncode == 0
