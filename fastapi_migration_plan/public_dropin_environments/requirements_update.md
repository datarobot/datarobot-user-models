# Plan: Public Drop-in Environments Requirements Update

Update all public drop-in environments to include FastAPI/Uvicorn dependencies.

## Overview

All public drop-in environments that include Flask and Gunicorn in their `requirements.txt` need to be updated to also include FastAPI and Uvicorn for the migration period.

## Affected Files

| Environment | File Path |
|-------------|-----------|
| java_codegen | `public_dropin_environments/java_codegen/requirements.txt` |
| python3_keras | `public_dropin_environments/python3_keras/requirements.txt` |
| python3_onnx | `public_dropin_environments/python3_onnx/requirements.txt` |
| python3_pytorch | `public_dropin_environments/python3_pytorch/requirements.txt` |
| python3_sklearn | `public_dropin_environments/python3_sklearn/requirements.txt` |
| python3_xgboost | `public_dropin_environments/python3_xgboost/requirements.txt` |
| python311 | `public_dropin_environments/python311/requirements.txt` |
| r_lang | `public_dropin_environments/r_lang/requirements.txt` |
| vllm | `public_dropin_gpu_environments/vllm/requirements.txt` |
| nim_sidecar | `public_dropin_nim_environments/nim_sidecar/requirements.txt` |
| python311_notebook_base | `public_dropin_notebook_environments/python311_notebook_base/requirements.txt` |
| python311_notebook | `public_dropin_notebook_environments/python311_notebook/requirements.txt` |
| python39_notebook_gpu_rapids | `public_dropin_notebook_environments/python39_notebook_gpu_rapids/requirements.txt` |
| python39_notebook_gpu_tf | `public_dropin_notebook_environments/python39_notebook_gpu_tf/requirements.txt` |
| python39_notebook_gpu | `public_dropin_notebook_environments/python39_notebook_gpu/requirements.txt` |

## Required Changes

### Phase 1: Add FastAPI Dependencies (Dual Support)

Add the following packages to each `requirements.txt`:

```
# FastAPI and Uvicorn for new server implementation
fastapi>=0.100.0,<1.0.0
uvicorn[standard]>=0.23.0,<1.0.0
httpx>=0.24.0,<1.0.0
```

### Phase 2: After Flask Removal

Remove the following packages:

```
# Remove after Flask deprecation period
flask
gunicorn
gevent
werkzeug
uwsgi  # Notebook environments only
```

## Implementation Example

For `public_dropin_environments/python3_sklearn/requirements.txt`:

```diff
 # Core dependencies
 scikit-learn==1.x.x
 pandas==2.x.x
 numpy==1.x.x
 
 # Server dependencies
 flask==3.1.2
 gunicorn==23.0.0
+
+# FastAPI server (new)
+fastapi>=0.100.0,<1.0.0
+uvicorn[standard]>=0.23.0,<1.0.0
+httpx>=0.24.0,<1.0.0
```

## Timeline

| Phase | Duration | Action |
|-------|----------|--------|
| Phase 1 | Immediate | Add FastAPI dependencies to all environments |
| Phase 2 | After 6 months LTS | Remove Flask, Gunicorn, Gevent |

## Notes

- The `uvicorn[standard]` extra is important for performance (includes `uvloop` and `httptools`)
- `httpx` is required for async HTTP client functionality
- Ensure Python version in base images is 3.8+ as required by FastAPI/Uvicorn
- Some environments may have version conflicts with `pydantic` - audit and resolve
