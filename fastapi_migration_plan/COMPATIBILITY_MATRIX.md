# Compatibility Matrix: FastAPI Migration

This document defines the supported configurations and known compatibility issues.

---

## Python Version Support

| Python Version | Flask/Gunicorn | FastAPI/Uvicorn | Notes |
|----------------|----------------|-----------------|-------|
| 3.7 | ✅ Supported | ❌ Not supported | FastAPI requires 3.8+ |
| 3.8 | ✅ Supported | ✅ Supported | Minimum for FastAPI |
| 3.9 | ✅ Supported | ✅ Supported | |
| 3.10 | ✅ Supported | ✅ Supported | |
| 3.11 | ✅ Supported | ✅ Supported | Recommended |
| 3.12 | ✅ Supported | ✅ Supported | Best performance |

**Action Required:** 
- Environments using Python 3.7 must upgrade before using `DRUM_SERVER_TYPE=fastapi`
- Add runtime check in `run_uvicorn.py`:

```python
import sys
if sys.version_info < (3, 8):
    raise RuntimeError(
        "FastAPI server requires Python 3.8+. "
        "Current version: {}. Use DRUM_SERVER_TYPE=flask for Python 3.7.".format(
            sys.version
        )
    )
```

---

## Pydantic Version Compatibility

### The Problem

FastAPI has different version requirements depending on version:
- FastAPI < 0.100: Pydantic v1 (1.x)
- FastAPI >= 0.100: Pydantic v2 (2.x)

Many ML libraries still use Pydantic v1 internally.

### Compatibility Table

| FastAPI Version | Pydantic Version | ML Library Compatibility |
|-----------------|------------------|--------------------------|
| 0.95.x - 0.99.x | 1.10.x | ✅ Best compatibility |
| 0.100.x+ | 2.x | ⚠️ May conflict with some ML libs |

### Recommended Strategy

> ⚠️ **RECOMMENDATION: Start with Option A (Conservative) for initial migration, then upgrade to Option C after stabilization.**

**Option A: Use FastAPI 0.95-0.99 (Conservative) - RECOMMENDED FOR INITIAL MIGRATION**
```
# requirements.txt - Phase 1 (M1-M5)
fastapi>=0.95.0,<0.100.0
pydantic>=1.10.0,<2.0.0
uvicorn[standard]>=0.23.0,<1.0.0
```

✅ **Why this is recommended:**
- Maximum compatibility with existing ML ecosystem (langchain, mlflow, ray)
- No risk of breaking customer models that depend on Pydantic v1
- Stable, battle-tested versions
- Easy upgrade path to v2 later

**Option B: Use FastAPI 0.100+ with Pydantic v1 compatibility layer**
```
# requirements.txt - Only if absolutely needed
fastapi>=0.100.0,<0.110.0
pydantic>=2.0.0,<3.0.0

# User code MUST migrate to pydantic.v1 shim:
# from pydantic.v1 import BaseModel
```

⚠️ **Use only if you need FastAPI 0.100+ features. Requires customer communication.**

**Option C: Use FastAPI 0.115+ (Latest) - FUTURE TARGET (after M6 stabilization)**
```
# requirements.txt - Phase 2 (post M6, ~6 months later)
fastapi>=0.115.0,<1.0.0
pydantic>=2.5.0,<3.0.0
```

📋 **Migration Timeline:**
| Phase | FastAPI Version | Pydantic | When |
|-------|----------------|----------|------|
| Initial (M1-M5) | 0.95-0.99 | 1.10.x | Start |
| Stable (M6) | 0.99.x | 1.10.x | After canary |
| Future | 0.115+ | 2.5+ | +6 months after M6 |

### Known Conflicts

| Library | Pydantic Requirement | Workaround |
|---------|---------------------|------------|
| `datarobot` SDK | v1 or v2 | Compatible with both |
| `langchain` < 0.1 | v1 only | Upgrade langchain or use Option A |
| `transformers` | No conflict | |
| `mlflow` < 2.9 | v1 only | Upgrade mlflow |
| `ray[serve]` < 2.8 | v1 only | Upgrade ray |

### Detection Script

```python
# scripts/check_pydantic_compat.py
"""Check for Pydantic version conflicts in the environment."""
import pkg_resources
import sys

def check_pydantic_compatibility():
    issues = []
    
    try:
        pydantic_version = pkg_resources.get_distribution("pydantic").version
        major = int(pydantic_version.split(".")[0])
    except pkg_resources.DistributionNotFound:
        print("Pydantic not installed")
        return
    
    print(f"Pydantic version: {pydantic_version}")
    
    # Known v1-only packages
    v1_only = {
        "langchain": "0.1.0",
        "mlflow": "2.9.0",
        "ray": "2.8.0",
    }
    
    for pkg, min_v2_compat in v1_only.items():
        try:
            installed = pkg_resources.get_distribution(pkg).version
            if major >= 2 and installed < min_v2_compat:
                issues.append(
                    f"⚠️  {pkg}=={installed} may not work with Pydantic v2. "
                    f"Upgrade to >={min_v2_compat}"
                )
        except pkg_resources.DistributionNotFound:
            pass
    
    if issues:
        print("\nCompatibility Issues Found:")
        for issue in issues:
            print(f"  {issue}")
        sys.exit(1)
    else:
        print("✅ No Pydantic compatibility issues detected")

if __name__ == "__main__":
    check_pydantic_compatibility()
```

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

### Core Dependencies (Phase 1: Conservative - Pydantic v1)

| Dependency | Minimum | Recommended | Maximum | Notes |
|------------|---------|-------------|---------|-------|
| fastapi | 0.95.0 | **0.99.1** | <0.100.0 | Last version with Pydantic v1 |
| uvicorn | 0.23.0 | 0.24.0 | <1.0.0 | Stable with graceful shutdown |
| httpx | 0.24.0 | 0.25.0 | <1.0.0 | For async HTTP |
| starlette | 0.27.0 | 0.32.0 | <0.33.0 | Compatible with FastAPI 0.99 |
| pydantic | 1.10.0 | 1.10.13 | <2.0.0 | v1 for compatibility |
| anyio | 3.0.0 | 3.7.1 | <4.0.0 | Stable v3 branch |

### Core Dependencies (Phase 2: Modern - Pydantic v2, post-stabilization)

| Dependency | Minimum | Recommended | Maximum | Notes |
|------------|---------|-------------|---------|-------|
| fastapi | 0.109.0 | 0.115.0 | <1.0.0 | Modern with Pydantic v2 |
| uvicorn | 0.27.0 | 0.30.0 | <1.0.0 | Latest features |
| httpx | 0.27.0 | 0.27.0 | <1.0.0 | For async HTTP |
| starlette | 0.36.0 | 0.37.0 | <1.0.0 | Latest compatible |
| pydantic | 2.5.0 | 2.6.0 | <3.0.0 | v2 with performance gains |
| anyio | 4.0.0 | 4.3.0 | <5.0.0 | Modern async support |

### Optional Dependencies

| Dependency | Purpose | When Required |
|------------|---------|---------------|
| uvloop | High-performance event loop | Linux/macOS production |
| orjson | Fast JSON serialization | High-throughput scenarios |
| python-multipart | Form data parsing | File uploads |

### Pinned Requirements

```
# requirements.txt - FastAPI dependencies
fastapi>=0.95.0,<1.0.0
uvicorn[standard]>=0.23.0,<1.0.0
httpx>=0.24.0,<1.0.0
python-multipart>=0.0.6
anyio>=3.0.0,<5.0.0

# Optional performance dependencies
uvloop>=0.17.0;platform_system!="Windows"
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

- [ ] Python version >= 3.8
- [ ] No Pydantic v1-only dependencies (or using compatibility layer)
- [ ] uvloop compatible (or using asyncio on Windows)
- [ ] Container runtime supports cgroups detection
- [ ] All dependencies at compatible versions
- [ ] Custom extensions migrated to `custom_fastapi.py`
