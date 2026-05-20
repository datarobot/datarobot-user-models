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

- Python 3.12 slim base
- `polars`, `pyarrow`, `datarobot`, `requests`, `httpx`
- A standalone `runner.py` that decodes a base64-encoded user snippet from
  `DR_SANDBOX_CODE_B64`, executes it, and prints a
  `__DR_SANDBOX_RESULT__:<json>` marker on stdout for the caller to parse.
- Runs as non-root UID 65534.

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
