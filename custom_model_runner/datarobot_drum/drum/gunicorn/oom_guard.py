#
#  Copyright 2026 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
"""Crash the gunicorn master when a worker is OOM-killed.

The kernel/cgroup OOM killer sends SIGKILL to the offending worker. By default
the gunicorn master just respawns it, so Kubernetes never observes a restart and
its OOM-handling (pod restart, backoff, alerts) never fires. DataRobot's runtime
delegates OOM handling to Kubernetes, so an absorbed OOM is a silent failure.

This module makes the master exit non-zero when a worker dies from an unsolicited
SIGKILL, letting the container exit and Kubernetes restart the pod.
"""
import logging
import os
import signal

from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

# Signals the master sends to its own workers on purpose (recycling, timeout,
# graceful shutdown). A SIGKILL that is NOT one of these was sent by something
# outside gunicorn — the OOM killer being the case we care about.
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
