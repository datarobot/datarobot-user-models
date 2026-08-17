#
#  Copyright 2026 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import os
import signal
from unittest.mock import MagicMock

import pytest

from datarobot_drum.drum.gunicorn import oom_guard


@pytest.fixture(autouse=True)
def _clean_state():
    oom_guard._sigkilled_workers.clear()
    oom_guard._master_initiated_kills.clear()
    yield
    oom_guard._sigkilled_workers.clear()
    oom_guard._master_initiated_kills.clear()


def _sigkill_status():
    return signal.SIGKILL


def test_sigkill_without_master_kill_is_oom():
    oom_guard._record_reaped(4242, _sigkill_status())
    assert oom_guard.is_oom_death(4242)


def test_normal_exit_is_not_oom():
    oom_guard._record_reaped(4242, 0)
    assert not oom_guard.is_oom_death(4242)


def test_non_kill_signal_is_not_oom():
    oom_guard._record_reaped(4242, signal.SIGTERM)
    assert not oom_guard.is_oom_death(4242)


def test_master_initiated_sigkill_is_not_oom():
    oom_guard._master_initiated_kills.add(4242)
    oom_guard._record_reaped(4242, _sigkill_status())
    assert not oom_guard.is_oom_death(4242)


def test_forget_clears_pid():
    oom_guard._record_reaped(4242, _sigkill_status())
    oom_guard.forget(4242)
    assert not oom_guard.is_oom_death(4242)


def test_handle_child_exit_halts_master_on_oom():
    oom_guard._record_reaped(4242, _sigkill_status())
    server = MagicMock()
    worker = MagicMock(pid=4242)

    oom_guard.handle_child_exit(server, worker)

    server.halt.assert_called_once()
    assert server.halt.call_args.kwargs["exit_status"] == 1


def test_handle_child_exit_ignores_normal_worker_death():
    oom_guard._record_reaped(4242, 0)
    server = MagicMock()
    worker = MagicMock(pid=4242)

    oom_guard.handle_child_exit(server, worker)

    server.halt.assert_not_called()


def test_handle_child_exit_forgets_pid_after_handling():
    oom_guard._record_reaped(4242, _sigkill_status())
    server = MagicMock()
    worker = MagicMock(pid=4242)

    oom_guard.handle_child_exit(server, worker)

    assert 4242 not in oom_guard._sigkilled_workers


def test_install_wraps_waitpid_and_records_sigkill(monkeypatch):
    fake_arbiter = MagicMock()
    fake_arbiter.Arbiter.kill_worker = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "gunicorn.arbiter", fake_arbiter)

    real_waitpid = os.waitpid
    monkeypatch.setattr(os, "waitpid", lambda *a, **k: (4242, signal.SIGKILL))
    try:
        oom_guard.install()
        pid, _ = os.waitpid(-1, 0)
    finally:
        os.waitpid = real_waitpid

    assert pid == 4242
    assert oom_guard.is_oom_death(4242)
