# Plan: Multi-Language Support in FastAPI

Support for R, Julia, and Java models in the FastAPI production server.

## Overview

DRUM supports multiple languages via specialized language predictors. When running with FastAPI, these predictors must be integrated correctly with the async lifecycle and thread pool.

## Language Integration Strategy

### 1. R and Julia (Python-bridged)

R and Julia models are typically loaded via Python-based predictors (`RPredictor`, `JuliaPredictor`) using `rpy2` or `julia` bridges.

**Strategy**:
- Treat these as synchronous models.
- Execute all prediction calls inside the `ThreadPoolExecutor` (already planned in `PredictionServer`).
- **Critical**: Ensure that the language bridge is thread-safe or that the pool size is restricted if the bridge has global locks (e.g., `rpy2` sometimes requires serial access).
- **Recommendation**: For R and Julia models, it is recommended to set `DRUM_FASTAPI_EXECUTOR_WORKERS=1` to avoid potential thread-safety issues with the underlying language bridges (`rpy2`, `PyJulia`).

**Action Items**:
- Add `DRUM_FASTAPI_EXECUTOR_WORKERS=1` as a default recommendation for R/Julia if thread safety issues are detected.
- Test with standard R/Julia model templates.

### 2. Java (Sidecar or Py4J)

Java models often run in a separate process or via Py4J bridge.

**Strategy**:
- For Py4J, execute calls in the thread pool.
- For unstructured Java models (running as a sidecar), ensure the proxy logic in `PredictionServer` correctly handles the connection to the Java sidecar.

## Verification Tests

### R Language Parity
```bash
MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE=fastapi \
drum server --code-dir model_templates/r_lang --address localhost:8080 --target-type regression
```

### Julia Language Parity
```bash
MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE=fastapi \
drum server --code-dir model_templates/julia --address localhost:8080 --target-type regression
```

### Java Language Parity
```bash
MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE=fastapi \
drum server --code-dir model_templates/java_codegen --address localhost:8080 --target-type regression
```

## Key Considerations

- **Process Lifecycle**: The Java sidecar process (if any) must be registered with `FastAPIWorkerCtx` for proper termination on shutdown.
- **Error Propagation**: Ensure that errors from the language-specific side (e.g., a JVM crash) are caught by the FastAPI error handlers and return a 513/500 status code.
- **Resource Usage**: Measure memory usage of bridged environments (R/Julia) as they might behave differently under the Uvicorn worker model.
