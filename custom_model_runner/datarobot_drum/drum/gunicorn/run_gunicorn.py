import logging
from pathlib import Path
import sys
import os
import shlex

from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


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

    package_dir_str = str(package_dir)
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if current_pythonpath:
        os.environ["PYTHONPATH"] = f"{package_dir_str}{os.pathsep}{current_pythonpath}"
    else:
        os.environ["PYTHONPATH"] = package_dir_str

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

    # Replace this process with gunicorn so its master runs as PID 1, receiving container signals and owning the exit status directly.
    try:
        os.execve(sys.executable, gunicorn_command, os.environ)
    except FileNotFoundError:
        logger.error("gunicorn module not found. Ensure it is installed.")
        raise


if __name__ == "__main__":
    main_gunicorn()
