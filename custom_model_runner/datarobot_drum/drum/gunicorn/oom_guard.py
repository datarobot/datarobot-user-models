#
#  Copyright 2026 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
"""Crash the gunicorn master when a worker is OOM-killed.

The OOM killer SIGKILLs a worker and the master silently respawns it, so
Kubernetes never sees the restart and its OOM handling never fires. This makes
the master exit non-zero on an unsolicited worker SIGKILL so k8s restarts the pod.
"""
import logging
import os
import signal

from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

# Kills the master issued itself; a SIGKILL not from here came from the OOM killer.
_master_initiated_kills = set()

# PIDs the master reaped that were terminated by an unsolicited SIGKILL.
_sigkilled_workers = set()


def _record_reaped(pid, status):
    if pid > 0 and os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL:
        _sigkilled_workers.add(pid)


def is_oom_death(pid):
    """A worker died from OOM if it was SIGKILLed without the master asking for it."""
    return pid in _sigkilled_workers and pid not in _master_initiated_kills


def forget(pid):
    _sigkilled_workers.discard(pid)
    _master_initiated_kills.discard(pid)


def install():
    """Wire OOM detection into the running master process.

    Wraps ``os.waitpid`` (the only place the master reaps workers) to capture the
    exit status the ``child_exit`` hook is never given, and ``Arbiter.kill_worker``
    to remember kills the master issued itself so they are not mistaken for OOM.
    """
    real_waitpid = os.waitpid

    def waitpid(*args, **kwargs):
        pid, status = real_waitpid(*args, **kwargs)
        try:
            _record_reaped(pid, status)
        except Exception:
            pass
        return pid, status

    os.waitpid = waitpid

    from gunicorn.arbiter import Arbiter

    real_kill_worker = Arbiter.kill_worker

    def kill_worker(self, pid, sig):
        # Only a master-sent SIGKILL can be confused with an OOM SIGKILL; other
        # signals never surface as WTERMSIG==SIGKILL, so tracking them would only
        # falsely mark live workers (e.g. SIGUSR1 log-reopen) and disable the guard.
        if sig == signal.SIGKILL:
            _master_initiated_kills.add(pid)
        return real_kill_worker(self, pid, sig)

    Arbiter.kill_worker = kill_worker


def handle_child_exit(server, worker):
    """``child_exit`` gunicorn hook: halt the master if the worker was OOM-killed."""
    pid = worker.pid
    try:
        if is_oom_death(pid):
            logger.error(
                "Worker (pid:%s) was killed by SIGKILL (out of memory); halting gunicorn "
                "master with a non-zero status so Kubernetes restarts the pod instead of "
                "silently respawning the worker.",
                pid,
            )
            server.halt(reason="worker killed by OOM", exit_status=1)
    finally:
        forget(pid)
