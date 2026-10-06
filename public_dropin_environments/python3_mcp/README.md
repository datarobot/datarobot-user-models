# Python 3 MCP Drop-In Template Environment

A serving-only execution environment for MCP (Model Context Protocol) servers built
with [FastMCP](https://gofastmcp.com/) and the DataRobot MCP toolkit
(`datarobot-genai[drmcp]`).

It is a deliberate narrowing of [`python311_genai_agents`](../python311_genai_agents):
same base image, same uv/lock toolchain — but only the MCP dependency line, and none of
the Codespaces/Notebooks runtime. There is no sshd, no Jupyter kernel gateway, no
IPython extensions, no monitoring agent and no DataRobot CLI, and the only port exposed
is 8080.

The environment name is deliberately version-less (precedent:
[`python3_genai_agents`](../python3_genai_agents)): the interpreter inside may move
within the Python 3 line (3.12 today) without renaming the environment or breaking
consumers that reference it by name.

This is not a replacement for
[`dr_mcp_execute_sandbox_minimal`](../dr_mcp_execute_sandbox_minimal), which is a
short-lived sandbox for the `execute_code` MCP *tool*. This environment *hosts* an MCP
server.

## Supported libraries

For specific version information and the complete list of included packages, see
[pyproject.toml](pyproject.toml) and [uv.lock](uv.lock). `requirements.txt` is a
generated two-line summary whose only job is to render the package list in the
Execution Environment UI — never edit it by hand (see [update_deps.sh](update_deps.sh)).

## Deployment surfaces

### Custom model

The platform runs `/opt/code/start_server.sh` **from the model bundle** — this image
deliberately ships no start script of its own, so the bundle's script (and therefore the
bundle's own dependency lock) is always authoritative. Package your server with:

```
app/
  main.py          # must be runnable as `python -m app.main`
pyproject.toml
uv.lock
start_server.sh    # the af-component-datarobot-mcp template ships the reference script
```

The reference `start_server.sh` finds the venv baked into this image via `$VENV_DIR`,
delta-`uv sync --frozen`s the bundle's lock into it (resolving from the uv cache warmed
at image build time — near-instant when the bundle's versions match the baked ones), and
execs `python -m app.main`. Because the sync runs against the *bundle's* lock, a bundle
pinning a newer `datarobot-genai` than the baked one still runs its own version. Routes
are automatically mounted under the deployment prefix — `drmcp` reads `URL_PREFIX` from
the environment itself (`MCPServerConfig.mount_path`), so nothing needs to be passed to
the server.

### Workload API (code-to-workload)

This image is used as the **base image for a generated Dockerfile** that copies the
bundle and sets its own entrypoint and readiness probe. Three properties of the image
make that work, and all three are load-bearing:

1. `$VENV_PATH/bin` is on `PATH` at image level, so an entrypoint like
   `["python", "-m", "app.main"]` resolves without sourcing an activate script.
2. There is **no `ENTRYPOINT`** in this image — an inherited entrypoint would be
   prefixed to the generated one and fail immediately.
3. `HOME`, the uv cache and the venv are writable by UID 1000, and nothing depends on
   `CODE_DIR` being populated.

The generated build does **not** install the bundle's dependencies itself, so an
entrypoint of `["python", "-m", "app.main"]` serves exactly the baked package set. To
make the bundle's lock authoritative on this surface too, use
`["sh", "start_server.sh"]` as the entrypoint (the af-component-datarobot-mcp infra
does) — it delta-syncs the bundle's lock into the baked venv before starting the server.

## Build locally

1. From the terminal, run
   `tar -czvf py_dropin.tar.gz -C /path/to/public_dropin_environments/python3_mcp/ .`
2. Using either the API or the UI, create a new Custom Environment with the tarball
   created in step 1.

## Keeping dependencies fresh

`datarobot-genai` releases several times a week. The pyproject pins a **range**
(`>=x.y.z,<x.y+1.0`), so a patch-level release needs no pyproject edit — running

```bash
./update_deps.sh    # requires uv >= 0.10.0, asserted at image build time
```

is the entire refresh: it runs `uv lock --upgrade`, regenerates `requirements.txt`, and
bumps `environmentVersionId` (the version id is the image tag — an unbumped id is never
rebuilt or installed). Wire it to a scheduled pipeline to keep the baked venv warm
without hand-made PRs; even when the bake lags, the runtime sync above keeps deployed
bundles on their own pinned versions.

A **minor** (`y`) bump in `datarobot-genai` is a breaking change: it requires a
deliberate range edit here **and** in `af-component-datarobot-mcp`'s template
`pyproject.toml.jinja`, moved in lockstep.

CVE floors live between the `cve-sync:begin` / `cve-sync:end` markers in
`pyproject.toml` and are owned by cve-sync policy, not by this repo.
