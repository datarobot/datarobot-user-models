# Python 3.11 UV Environment

Base execution environment for Code-to-Workload (C2W) deployments using Python 3.11 and the [uv](https://github.com/astral-sh/uv) package manager.

## What's Included

- Python 3.11
- uv package manager
- Build tools for compiling native Python extensions

## What's NOT Included

- No pre-installed Python packages (no DRUM, no ML libraries)
- No `start_server.sh` entrypoint — C2W provides its own entrypoint
- No `requirements.txt` — user dependencies are installed at build time via `uv sync`

## How It Works

This image serves as a base layer for C2W-generated Dockerfiles. The C2W build process:

1. Uses this image as `FROM` base
2. Copies the user's `pyproject.toml` and `uv.lock`
3. Runs `uv sync` to install dependencies
4. Copies user code
5. Sets the user-defined entrypoint

## Local Development

Build the local image:

```bash
docker build -f Dockerfile.local -t python311-uv-local .
```

Verify:

```bash
docker run --rm python311-uv-local python --version
docker run --rm python311-uv-local uv --version
```
