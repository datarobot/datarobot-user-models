import logging
import signal
import subprocess
from pathlib import Path
import sys
import os
import shlex

from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


# Signals the orchestrator/runtime may send to the drum entry-point that must
# reach the gunicorn master so it can run its graceful shutdown (which is what
# fires each worker's worker_exit hook → ctx.stop() / ctx.cleanup()).
_FORWARDED_SIGNALS = (
    signal.SIGTERM,
    signal.SIGINT,
    signal.SIGHUP,
    signal.SIGQUIT,
    signal.SIGUSR1,
    signal.SIGUSR2,
)


def _install_signal_forwarder(proc):
    def _forward(sig, _frame):
        logger.info("Forwarding signal %s to gunicorn master (pid=%s)", sig, proc.pid)
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass

    for s in _FORWARDED_SIGNALS:
        try:
            signal.signal(s, _forward)
        except (ValueError, OSError):
            # ValueError: not the main thread; OSError: signal not available
            # on this platform. Skip and let the default disposition stand.
            pass


def main_gunicorn():
    # Resolve directory containing this script so we can always find config
    package_dir = Path(__file__).resolve().parent
    config_path = package_dir / "gunicorn.conf.py"

    if not config_path.is_file():
        raise FileNotFoundError(f"Gunicorn config not found: {config_path}")

    # Export all provided CLI args (excluding script) into DRUM_GUNICORN_DRUM_ARGS
    extra_args = sys.argv
    if extra_args:
        try:
            os.environ["DRUM_GUNICORN_DRUM_ARGS"] = shlex.join(extra_args)
        except AttributeError:
            os.environ["DRUM_GUNICORN_DRUM_ARGS"] = " ".join(shlex.quote(a) for a in extra_args)

    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        env["PYTHONPATH"] = f"{package_dir}{os.pathsep}{current_pythonpath}"
    else:
        env["PYTHONPATH"] = str(package_dir)

    # Use the gunicorn module explicitly to avoid issues where a shadowed
    # console script named "gunicorn" actually invokes the DRUM CLI.
    gunicorn_command = [
        sys.executable,
        "-m",
        "gunicorn",
        "-c",
        str(config_path),
        "app:app",  # module:variable; app.py sits next to this script
    ]

    try:
        proc = subprocess.Popen(gunicorn_command, env=env)
    except FileNotFoundError:
        logger.error("gunicorn module not found. Ensure it is installed.")
        raise

    # Without this, a SIGTERM from the container runtime kills this Python
    # parent immediately and leaves the gunicorn master orphaned — worker_exit
    # callbacks (ctx.stop()/ctx.cleanup()) never run.
    _install_signal_forwarder(proc)

    returncode = proc.wait()
    if returncode != 0:
        logger.error("Gunicorn exited with non-zero status %s", returncode)
        sys.exit(returncode)


if __name__ == "__main__":
    main_gunicorn()
