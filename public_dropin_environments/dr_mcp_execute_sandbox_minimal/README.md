# dr_mcp_execute_sandbox_minimal

Minimal Python container image used by the **`execute_code`** MCP tool in
[datarobot-oss/datarobot-genai](https://github.com/datarobot-oss/datarobot-genai)
(`src/datarobot_genai/drtools/sandbox/`).

Unlike the other entries in `public_dropin_environments/`, this is **not** a
custom-model drop-in environment — it's a runtime image submitted to the
DataRobot Workload API as a single-container, short-lived sandbox for
executing untrusted Python code on behalf of MCP clients. It is not
registered in the DataRobot environment catalog.

## What it ships

- Chainguard `python-fips:3.12` runtime (multi-stage build from `:3.12-dev`)
- `polars`, `pyarrow`, `datarobot`, `requests`, `httpx`
- A standalone `runner.py` that decodes a base64-encoded user snippet from
  `DR_SANDBOX_CODE_B64`, executes it, and prints a
  `__DR_SANDBOX_RESULT__:<json>` marker on stdout for the caller to parse.
- Runs as non-root UID 65534.

## ⏱️ Execution timeout — capped at 1 hour

**User code executed inside this image is hard-capped at 1 hour (3600s) of
wall-clock time.** After the timeout, `runner.py` interrupts the snippet,
exits with code `124`, and still emits the `__DR_SANDBOX_RESULT__:null`
marker so callers don't hang waiting for output.

- The cap is enforced in `runner.py` via `SIGALRM` and is controlled by
  the `DR_SANDBOX_TIMEOUT_SECS` env var (default `3600`). Set it lower
  to shorten the cap for a given run; set it to `0` to disable the
  in-process timer entirely.
- This in-process timeout is **defense-in-depth for accidental hangs only**
  — it is **not** a security boundary. Malicious user code can reset the
  `SIGALRM` handler. The real wall-clock enforcement lives on the caller
  side (`DataRobotWorkloadSandbox` / Workload API), which may impose its
  own (lower) cap that this runner cannot extend.
- If you need longer-than-1-hour execution: open an issue and discuss
  raising the default cap, or override `DR_SANDBOX_TIMEOUT_SECS` at the
  caller — but check that the caller / Workload API cap allows it.

## Published image

Built and pushed by Harness on master push when files in this folder
change. The published URI is:

```
datarobotdev/datarobot-user-models:public_dropin_environments_dr_mcp_execute_sandbox_minimal_latest
```

## Source of truth

The Dockerfile and runner are mirrored from
[`docker/sandbox/`](https://github.com/datarobot-oss/datarobot-genai/tree/main/docker/sandbox)
in `datarobot-oss/datarobot-genai`. When updating, change both and bump the
version in `pyproject.toml` there.
