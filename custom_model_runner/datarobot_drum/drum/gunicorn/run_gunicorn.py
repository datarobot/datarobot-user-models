import subprocess
from pathlib import Path
import sys


def main_gunicorn():
    # Resolve directory containing this script so we can always find config and app.py
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "gunicorn.conf.py"

    if not config_path.is_file():
        raise FileNotFoundError(f"Gunicorn config not found: {config_path}")

    gunicorn_command = [
        "gunicorn",
        "-c",
        str(config_path),
        "app:app",  # module:variable; app.py sits next to this script
    ]

    try:
        # Run with cwd set so module "app" is importable even if caller runs from elsewhere
        subprocess.run(gunicorn_command, cwd=base_dir, check=True)
    except FileNotFoundError as e:
        print("gunicorn executable not found. Ensure it is installed and on PATH.", file=sys.stderr)
        raise
    except subprocess.CalledProcessError as e:
        print(f"Gunicorn exited with non-zero status {e.returncode}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main_gunicorn()
