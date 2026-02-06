# Compatibility Matrix: FastAPI Migration

This document defines the supported configurations and known compatibility issues.

---

## Python Version Support

| Python Version | Flask/Gunicorn | FastAPI/Uvicorn | Notes |
|----------------|----------------|-----------------|-------|
| 3.7 | ✅ Supported | ❌ Not supported | EOL, use Flask |
| 3.8 | ✅ Supported | ❌ Not supported | Use Flask for 3.8 |
| 3.9 | ✅ Supported | ✅ Supported | Minimum for FastAPI |
| 3.10 | ✅ Supported | ✅ Supported | |
| 3.11 | ✅ Supported | ✅ Supported | Recommended |
| 3.12 | ✅ Supported | ✅ Supported | Best performance |

**Action Required:** 
- Environments using Python 3.8 or lower must use `DRUM_SERVER_TYPE=flask`
- Python 3.12 requires `uvloop>=0.19.0` to avoid segfaults
- Add runtime check in `run_uvicorn.py`:

```python
import sys
if sys.version_info < (3, 9):
    raise RuntimeError(
        f"FastAPI server requires Python 3.9+. "
        f"Current version: {sys.version}. Use DRUM_SERVER_TYPE=flask for Python 3.8."
    )
```

---

## Pydantic Version Compatibility

### Strategy: Immediate Pydantic v2 Support

We use Pydantic v2 from the start for:
- **Performance:** 5-50x faster validation
- **Simpler code:** No v1 compatibility layer needed
- **Future-proof:** FastAPI 1.0 will require Pydantic v2
- **Better typing:** Improved type hints and IDE support
- **Python 3.12:** Full compatibility with latest Python

### Required Versions

```
# requirements.txt
fastapi>=0.109.0,<1.0.0
pydantic>=2.5.0,<3.0.0
uvicorn[standard]>=0.27.0,<1.0.0
uvloop>=0.19.0;platform_system!="Windows"
```

### Compatibility Table

| FastAPI Version | Pydantic Version | Status |
|-----------------|------------------|--------|
| >=0.109.0,<1.0.0 | >=2.5.0,<3.0.0 | **Default** |

### Known Library Compatibility

If using these libraries in custom models, ensure minimum versions:

| Library | Minimum Version | Pydantic v2 Support | Notes |
|---------|-----------------|---------------------|-------|
| `datarobot` SDK | Any | ✅ Yes | Compatible with both v1 and v2 |
| `langchain` | >= 0.1.0 | ✅ Yes | Requires upgrade from older versions |
| `mlflow` | >= 2.9.0 | ✅ Yes | Requires upgrade from older versions |
| `ray[serve]` | >= 2.8.0 | ✅ Yes | Requires upgrade from older versions |
| `transformers` | Any | ✅ Yes | No conflict |

### Detection Script

```python
# scripts/check_pydantic_compat.py
"""Check for Pydantic v2 compatibility in the environment."""
from importlib.metadata import version, PackageNotFoundError
import sys

def check_pydantic_compatibility():
    issues = []
    
    try:
        pydantic_version = version("pydantic")
        major = int(pydantic_version.split(".")[0])
    except PackageNotFoundError:
        print("Pydantic not installed")
        return
    
    print(f"Pydantic version: {pydantic_version}")
    
    if major < 2:
        issues.append(
            f"⚠️  Pydantic {pydantic_version} is v1. DRUM FastAPI server requires Pydantic v2. "
            f"Upgrade with: pip install 'pydantic>=2.5.0'"
        )
    
    # Known packages that require minimum versions for Pydantic v2
    v2_min_versions = {
        "langchain": "0.1.0",
        "mlflow": "2.9.0",
        "ray": "2.8.0",
    }
    
    for pkg, min_v2_compat in v2_min_versions.items():
        try:
            installed = version(pkg)
            if installed < min_v2_compat:
                issues.append(
                    f"⚠️  {pkg}=={installed} may not work with Pydantic v2. "
                    f"Upgrade to >={min_v2_compat}"
                )
        except PackageNotFoundError:
            pass
    
    if issues:
        print("\nCompatibility Issues Found:")
        for issue in issues:
            print(f"  {issue}")
        sys.exit(1)
    else:
        print("✅ No Pydantic v2 compatibility issues detected")

if __name__ == "__main__":
    check_pydantic_compatibility()
```

### Pydantic v1 to v2 Migration Patterns

If your custom model code uses Pydantic v1, update using these patterns:

| v1 Pattern | v2 Pattern |
|------------|------------|
| `class Config:` | `model_config = ConfigDict(...)` |
| `.dict()` | `.model_dump()` |
| `.json()` | `.model_dump_json()` |
| `@validator` | `@field_validator` |
| `@root_validator` | `@model_validator` |
| `constr(regex=...)` | `Annotated[str, Field(pattern=...)]` |
| `constr(min_length=1)` | `Annotated[str, Field(min_length=1)]` |
| `conlist(Item, min_items=1)` | `Annotated[List[Item], Field(min_length=1)]` |
| `.parse_raw()` | `.model_validate_json()` |
| `.parse_obj()` | `.model_validate()` |
| `__fields__` | `model_fields` |
| `.schema()` | `.model_json_schema()` |

---

## Operating System Support

| OS | Flask/Gunicorn | FastAPI/Uvicorn | uvloop Support |
|----|----------------|-----------------|----------------|
| Linux (x86_64) | ✅ | ✅ | ✅ |
| Linux (ARM64) | ✅ | ✅ | ✅ |
| macOS (Intel) | ✅ | ✅ | ✅ |
| macOS (Apple Silicon) | ✅ | ✅ | ✅ |
| Windows | ✅ | ✅ | ❌ Not available |
| Alpine Linux | ✅ | ✅ | ⚠️ Requires musl-compatible build |

### Windows-Specific Configuration

```python
# In config.py
import platform

@dataclass
class UvicornConfig:
    loop: str = "auto"
    
    @classmethod
    def from_runtime_params(cls) -> "UvicornConfig":
        config = cls()
        
        # Event loop implementation
        if RuntimeParameters.has("DRUM_UVICORN_LOOP"):
            requested_loop = str(RuntimeParameters.get("DRUM_UVICORN_LOOP")).lower()
            
            # uvloop not available on Windows
            if requested_loop == "uvloop" and platform.system() == "Windows":
                logger.warning(
                    "uvloop is not available on Windows. Falling back to asyncio."
                )
                config.loop = "asyncio"
            elif requested_loop in {"auto", "asyncio", "uvloop"}:
                config.loop = requested_loop
        else:
            # Auto-detect best loop for platform
            if platform.system() == "Windows":
                config.loop = "asyncio"
            else:
                config.loop = "auto"  # Will use uvloop if available
        
        return config
```

---

## Container Runtime Support

| Container Runtime | Supported | Notes |
|-------------------|-----------|-------|
| Docker | ✅ | Recommended |
| containerd | ✅ | |
| Podman | ✅ | |
| CRI-O | ✅ | |

### Cgroups v1 vs v2

| Cgroup Version | Memory Detection | Notes |
|----------------|------------------|-------|
| v1 | `/sys/fs/cgroup/memory/memory.limit_in_bytes` | Legacy, Docker default on older systems |
| v2 | `/sys/fs/cgroup/memory.max` | Modern, K8s 1.25+, Docker 20.10+ |
| v2 (nested) | `/sys/fs/cgroup/<slice>/memory.max` | Systemd-managed containers |

```python
# In resource_monitor.py
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Unlimited memory sentinel values
CGROUP_V1_UNLIMITED = 9223372036854771712  # 2^63 - 4096
CGROUP_V2_UNLIMITED = "max"


def _find_cgroup_v2_memory_path() -> Optional[Path]:
    """
    Find the correct cgroup v2 memory path, handling nested cgroups.
    
    Kubernetes and systemd may place containers in nested cgroup hierarchies:
    - /sys/fs/cgroup/memory.max (root cgroup)
    - /sys/fs/cgroup/system.slice/docker-xxx.scope/memory.max (nested)
    - /sys/fs/cgroup/kubepods.slice/kubepods-burstable.slice/.../memory.max (K8s)
    """
    # First, check if we're in a cgroup v2 environment
    cgroup_root = Path("/sys/fs/cgroup")
    
    # Check root cgroup first
    root_memory = cgroup_root / "memory.max"
    if root_memory.exists():
        try:
            value = root_memory.read_text().strip()
            if value != CGROUP_V2_UNLIMITED:
                return root_memory
        except (IOError, PermissionError):
            pass
    
    # Try to find our cgroup from /proc/self/cgroup
    try:
        cgroup_info = Path("/proc/self/cgroup").read_text()
        for line in cgroup_info.strip().split("\n"):
            parts = line.split(":")
            if len(parts) >= 3:
                # cgroup v2 format: 0::/path
                if parts[0] == "0" and parts[1] == "":
                    cgroup_path = cgroup_root / parts[2].lstrip("/")
                    memory_path = cgroup_path / "memory.max"
                    if memory_path.exists():
                        return memory_path
    except (IOError, PermissionError):
        pass
    
    return None


def _find_cgroup_v1_memory_path() -> Optional[Path]:
    """Find cgroup v1 memory path."""
    paths = [
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        # Docker-specific paths
        Path("/sys/fs/cgroup/memory/docker/memory.limit_in_bytes"),
    ]
    
    # Also try to find from /proc/self/cgroup
    try:
        cgroup_info = Path("/proc/self/cgroup").read_text()
        for line in cgroup_info.strip().split("\n"):
            parts = line.split(":")
            if len(parts) >= 3 and "memory" in parts[1]:
                cgroup_path = Path("/sys/fs/cgroup/memory") / parts[2].lstrip("/")
                memory_path = cgroup_path / "memory.limit_in_bytes"
                if memory_path.exists():
                    paths.insert(0, memory_path)
                break
    except (IOError, PermissionError):
        pass
    
    for path in paths:
        if path.exists():
            return path
    
    return None


def detect_memory_limit() -> int:
    """
    Detect container memory limit, supporting both cgroups v1 and v2.
    
    Handles:
    - Root cgroups
    - Nested cgroups (Kubernetes, systemd)
    - Docker containers
    - Hybrid v1/v2 setups
    
    Returns:
        Memory limit in bytes, or system total if unlimited/unavailable.
    """
    import psutil
    
    # Try cgroups v2 first (modern)
    cgroup_v2_path = _find_cgroup_v2_memory_path()
    if cgroup_v2_path:
        try:
            value = cgroup_v2_path.read_text().strip()
            if value != CGROUP_V2_UNLIMITED:
                limit = int(value)
                logger.debug("Detected cgroup v2 memory limit: %d bytes from %s", limit, cgroup_v2_path)
                return limit
        except (IOError, ValueError, PermissionError) as e:
            logger.warning("Failed to read cgroup v2 memory limit: %s", e)
    
    # Try memory.high as soft limit indicator (cgroup v2)
    if cgroup_v2_path:
        high_path = cgroup_v2_path.parent / "memory.high"
        if high_path.exists():
            try:
                value = high_path.read_text().strip()
                if value != CGROUP_V2_UNLIMITED:
                    limit = int(value)
                    logger.debug("Detected cgroup v2 memory.high: %d bytes", limit)
                    # Note: memory.high is soft limit, memory.max is hard limit
                    # We prefer .max but .high is useful for monitoring
            except (IOError, ValueError):
                pass
    
    # Fall back to cgroups v1
    cgroup_v1_path = _find_cgroup_v1_memory_path()
    if cgroup_v1_path:
        try:
            value = int(cgroup_v1_path.read_text().strip())
            if value < CGROUP_V1_UNLIMITED:
                logger.debug("Detected cgroup v1 memory limit: %d bytes from %s", value, cgroup_v1_path)
                return value
        except (IOError, ValueError, PermissionError) as e:
            logger.warning("Failed to read cgroup v1 memory limit: %s", e)
    
    # Fall back to system memory
    system_memory = psutil.virtual_memory().total
    logger.debug("No cgroup memory limit detected, using system memory: %d bytes", system_memory)
    return system_memory
```

### Cgroup Detection Testing

```bash
# Check which cgroup version is in use
if [ -f /sys/fs/cgroup/cgroup.controllers ]; then
    echo "cgroup v2 (unified)"
else
    echo "cgroup v1 (legacy) or hybrid"
fi

# Find current container's cgroup
cat /proc/self/cgroup

# Check memory limit (v2)
cat /sys/fs/cgroup/memory.max 2>/dev/null || echo "Not v2 root"

# Check memory limit (v1)
cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo "Not v1"
```

---

## Dependency Version Matrix

### Core Dependencies (Pydantic v2)

| Dependency | Minimum | Recommended | Maximum | Notes |
|------------|---------|-------------|---------|-------|
| fastapi | 0.109.0 | **0.115.0** | <1.0.0 | Modern with Pydantic v2 |
| uvicorn | 0.27.0 | 0.30.0 | <1.0.0 | Latest features |
| httpx | 0.27.0 | 0.27.0 | <1.0.0 | For async HTTP |
| starlette | 0.36.0 | 0.37.0 | <1.0.0 | Latest compatible |
| pydantic | 2.5.0 | 2.6.0 | <3.0.0 | v2 with 5-50x performance gains |
| anyio | 4.0.0 | 4.3.0 | <5.0.0 | Modern async support |

### Optional Dependencies

| Dependency | Purpose | Version | Notes |
|------------|---------|---------|-------|
| uvloop | High-performance event loop | >=0.19.0 | **Critical:** >=0.19.0 for Python 3.12 |
| orjson | Fast JSON serialization | >=3.9.0 | High-throughput scenarios |
| python-multipart | Form data parsing | >=0.0.6 | File uploads |

### Pinned Requirements

```
# requirements.txt - FastAPI dependencies (Pydantic v2)
fastapi>=0.109.0,<1.0.0
uvicorn[standard]>=0.27.0,<1.0.0
starlette>=0.36.0,<1.0.0
pydantic>=2.5.0,<3.0.0
httpx>=0.27.0,<1.0.0
python-multipart>=0.0.6
anyio>=4.0.0,<5.0.0

# Event loop - critical for Python 3.12
uvloop>=0.19.0;platform_system!="Windows"

# Optional performance dependencies
orjson>=3.9.0
```

---

## Known Issues and Workarounds

### Issue 1: `anyio` Version Conflicts

**Symptom:** `ImportError: cannot import name 'run_sync' from 'anyio'`

**Cause:** Conflicting anyio versions between packages

**Solution:**
```bash
pip install "anyio>=3.0.0,<5.0.0"
```

### Issue 2: `httptools` Build Failures on Alpine

**Symptom:** `error: command 'gcc' failed with exit status 1`

**Solution:**
```dockerfile
# Alpine Dockerfile
RUN apk add --no-cache gcc musl-dev python3-dev
RUN pip install httptools --no-binary httptools
```

### Issue 3: `uvloop` Segfault on Python 3.12

**Symptom:** Segmentation fault on startup with uvloop

**Solution:** Update uvloop to >= 0.19.0
```bash
pip install "uvloop>=0.19.0"
```

### Issue 4: SSL Certificate Verification with httpx

**Symptom:** `httpx.ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED]`

**Solution:** Ensure CA certificates are available
```dockerfile
# Dockerfile
RUN apt-get update && apt-get install -y ca-certificates
# Or for custom CA:
ENV SSL_CERT_FILE=/path/to/ca-bundle.crt
```

---

## Migration Compatibility Checklist

Before enabling `DRUM_SERVER_TYPE=fastapi`:

- [ ] Python version >= 3.9 (3.11 or 3.12 recommended)
- [ ] Pydantic v2 compatible (>= 2.5.0)
- [ ] uvloop >= 0.19.0 on Python 3.12 (or using asyncio on Windows)
- [ ] ML library versions compatible with Pydantic v2:
  - [ ] langchain >= 0.1.0 (if used)
  - [ ] mlflow >= 2.9.0 (if used)
  - [ ] ray[serve] >= 2.8.0 (if used)
- [ ] Container runtime supports cgroups detection
- [ ] All dependencies at compatible versions
- [ ] Custom model code migrated from Pydantic v1 to v2 patterns
- [ ] Custom extensions migrated to `custom_fastapi.py`
