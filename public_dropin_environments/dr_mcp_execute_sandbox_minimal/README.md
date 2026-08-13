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
- `folium`, for the `create_chart_panel` MCP tool's map charts — see
  [Why folium and no other charting library](#why-folium-and-no-other-charting-library)
- A standalone `runner.py` that decodes a base64-encoded user snippet from
  `DR_SANDBOX_CODE_B64`, executes it, and prints a
  `__DR_SANDBOX_RESULT__:<json>` marker on stdout for the caller to parse.
- Runs as non-root UID 65534.

## Why folium and no other charting library

`create_chart_panel` (in datarobot-genai / wren-mcp) supports three charting
libraries, and only folium needs a package in this image:

| library  | user code returns             | needs a package here? |
| -------- | ----------------------------- | --------------------- |
| `plotly` | figure JSON dict              | no                    |
| `altair` | Vega-Lite spec dict           | no                    |
| `folium` | rendered leaflet HTML string  | **yes**               |

plotly and altair charts are plain JSON, so the tool has user code emit the
dict directly and never imports either package. Leaflet HTML is not
hand-authorable, so folium is installed and the map renders here.

Rendering folium *inside* the sandbox rather than in the calling MCP server is
deliberate. Passing a spec out for the server to render would mean either
dropping `style_function`/`highlight_function` — Python callables no JSON spec
can carry, and the whole point of a choropleth — or `eval`-ing a code string in
a shared, multi-tenant process. And `Map.render()` is a Jinja2 expansion that
materializes the entire feature set in memory, which wants a container with a
cap and a lifetime. It is also cheap: `folium`, `branca`, `jinja2`,
`markupsafe`, and `xyzservices` are pure Python and total ~0.4 MB, against the
~120 MB of numpy/pandas/polars/pyarrow already here (numpy and pandas arrive as
`datarobot` dependencies).

Before adding any further library, check whether its output format can be
emitted as JSON by user code instead — that keeps this image minimal.

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

> The publish trigger is path-filtered to this folder, so the `_latest` tag is
> (re)built only when files here change on master. Consumers pull the mutable
> `_latest` tag, so a merge here is what ships a dependency change — there is
> no version to bump on the caller side.

## Source of truth

This folder is the only home of the Dockerfile and runner. `datarobot-genai`
deliberately does **not** carry a copy: it shares just the wire contract
(`RESULT_MARKER`, the timeout exit code, and the marker parser) in
`src/datarobot_genai/drtools/core/sandbox/protocol.py`, precisely so a
hand-synced duplicate of the runner can't drift from the image. Change the
runner here, and update `protocol.py` only if the wire contract itself changes.
