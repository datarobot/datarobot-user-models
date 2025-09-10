import subprocess
from pathlib import Path
import sys
import os
import shlex


def main_gunicorn():
    # Resolve directory containing this script so we can always find config and app.py
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "gunicorn.conf.py"

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
        subprocess.run(gunicorn_command, cwd=base_dir, check=True)
    except FileNotFoundError:
        print("gunicorn module not found. Ensure it is installed.", file=sys.stderr)
        raise
    except subprocess.CalledProcessError as e:
        print(f"Gunicorn exited with non-zero status {e.returncode}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main_gunicorn()
