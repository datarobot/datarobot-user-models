import logging
import subprocess
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

    # Run Gunicorn from the model code directory so any relative paths (e.g. .deepeval)
    # are created under writable model code instead of inside site-packages.
    code_dir = Path(os.environ.get("CODE_DIR", "/opt/code"))

    if not config_path.is_file():
        raise FileNotFoundError(f"Gunicorn config not found: {config_path}")

    # Export all provided CLI args (excluding script) into DRUM_GUNICORN_DRUM_ARGS
    extra_args = sys.argv
    if extra_args:
        try:
            os.environ["DRUM_GUNICORN_DRUM_ARGS"] = shlex.join(extra_args)
        except AttributeError:
            os.environ["DRUM_GUNICORN_DRUM_ARGS"] = " ".join(shlex.quote(a) for a in extra_args)

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
        subprocess.run(gunicorn_command, cwd=code_dir, check=True)
    except FileNotFoundError:
        logger.error("gunicorn module not found. Ensure it is installed.")
        raise
    except subprocess.CalledProcessError as e:
        logger.error("Gunicorn exited with non-zero status %s", e.returncode)
        raise


if __name__ == "__main__":
    main_gunicorn()
