# Python 3 GenAI Agents Drop-In Template Environment

This environment runs GenAI agents as custom models and as agentic playground codespaces. It
ships no agent framework: the agent brings its own dependencies in `pyproject.toml` and
`uv.lock`, and its `run_agent.py` (playground) or `start_server.sh` (custom model) creates the
agent's virtualenv with `uv sync --frozen` at start. The image only provides Python, `uv`, the
Jupyter kernel and the monitoring agent that codespaces need.

This is a Python 3 base image. For the exact version of Python 3 currently used by the image please see the `Dockerfile` and `Dockerfile.local`. These pin specific python versions and Chainguard base images.

Additionally, this environment is fully compatible with `Codespaces` and `Notebooks` in the DataRobot platform.

## What the image provides

See [pyproject.toml](pyproject.toml). The `agentic_playground` extra is the Jupyter kernel
stack, pinned to what the notebook environments ship, plus `fastapi[all]` for the monitoring
agent in `agent/`. There are no runtime dependencies beyond that.

Agents rely on three world-writable directories baked into the image: `/opt/code` for the
agent's code, `/opt/venv` for the agent's virtualenv and `/tmp/uv-cache` for uv. Custom model
containers run as uid 1000, so these cannot be owned by the image user.

## Build locally

1. From the terminal, run `tar -czvf py_dropin.tar.gz -C /path/to/public_dropin_environments/python3_genai_agents/ .`
2. Using either the API or from the UI create a new Custom Environment with the tarball created in step 1.

_The Dockerfile.local should be used when customizing the Dockerfile or building locally._
When exporting a locally built image instead of a context, build it with
`--platform linux/amd64`; DataRobot nodes are amd64.
