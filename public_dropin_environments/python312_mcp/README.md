# Python 3.12 MCP Drop-In Template Environment

A serving-only execution environment for MCP (Model Context Protocol) servers built
with [FastMCP](https://gofastmcp.com/) and the DataRobot MCP toolkit
(`datarobot-genai[drmcp]`).

It is a deliberate narrowing of [`python311_genai_agents`](../python311_genai_agents):
same base image, same uv/lock toolchain, same custom-model start-script contract — but
only the MCP dependency line, and none of the Codespaces/Notebooks runtime. There is no
sshd, no Jupyter kernel gateway, no IPython extensions, no monitoring agent and no
DataRobot CLI, and the only port exposed is 8080.

This is not a replacement for
[`dr_mcp_execute_sandbox_minimal`](../dr_mcp_execute_sandbox_minimal), which is a
short-lived sandbox for the `execute_code` MCP *tool*. This environment *hosts* an MCP
server.

## Supported libraries

For specific version information and the complete list of included packages, see
[pyproject.toml](pyproject.toml) and [uv.lock](uv.lock).

## Deployment surfaces

### Custom model

The platform runs `/opt/code/start_server.sh` (this folder's
[`start_server.sh`](start_server.sh)). Package your server
with:

```
app/
  main.py          # must be runnable as `python -m app.main`
pyproject.toml
uv.lock
```

The script creates the venv at `$VENV_DIR`, `uv sync --frozen`s your project into it
(resolving from the uv cache baked at image build time), and execs `python -m app.main`.
Routes are automatically mounted under the deployment prefix — `drmcp` reads
`URL_PREFIX` from the environment itself, so nothing needs to be passed to the server.

### Workload API (code-to-workload)

This image is used as the **base image for a generated Dockerfile** that sets its own
entrypoint and readiness probe; `start_server.sh` is never invoked. Three properties of
the image make that work, and all three are load-bearing:

1. `$VENV_PATH/bin` is on `PATH` at image level, so an entrypoint like
   `["python", "-m", "app.main"]` resolves without sourcing an activate script.
2. There is **no `ENTRYPOINT`** in this image — an inherited entrypoint would be
   prefixed to the generated one and fail immediately.
3. `HOME`, the uv cache and the venv are writable by UID 1000, and nothing depends on
   `CODE_DIR` being populated.

## Build locally

1. From the terminal, run
   `tar -czvf py_dropin.tar.gz -C /path/to/public_dropin_environments/python312_mcp/ .`
2. Using either the API or the UI, create a new Custom Environment with the tarball
   created in step 1.

## [Development] Updating dependencies

`pyproject.toml`, `uv.lock` and `requirements.txt` are currently hand-maintained; unlike
`python311_genai_agents` there is no `af-component-agents` generator target for this
folder yet (tracked in [DESIGN.md](DESIGN.md) §5). To refresh the lock after editing
`pyproject.toml`:

```bash
uv lock          # requires uv >= 0.10.0, asserted at image build time
```

`requirements.txt` is a flat pinned list whose only job is to render the package list in
the Execution Environment UI. It carries no `datarobot-drum` line — this environment
ships no DRUM.

CVE floors live between the `cve-sync:begin` / `cve-sync:end` markers in
`pyproject.toml` and are owned by cve-sync policy, not by this repo. One pin currently
diverges from policy (`mcp>=1.28.1,<2.0.0`) and is documented in place.
