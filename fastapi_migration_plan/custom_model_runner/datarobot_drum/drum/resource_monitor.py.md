# Plan: custom_model_runner/datarobot_drum/drum/resource_monitor.py

Update `ResourceMonitor` to support Uvicorn process hierarchy and Cgroups v2 for memory statistics.

## Overview

The `ResourceMonitor` is used to report resource usage (CPU, Memory) in the `/stats/` endpoint. It needs to be updated to:
1. Correctly identify the DRUM parent process when running under Uvicorn.
2. Support Cgroups v2 for memory limit and usage discovery, as many modern container runtimes (like those on newer Kubernetes or macOS Docker Desktop) have transitioned to v2.

## Proposed Implementation

```python
"""
Resource Monitor updates for FastAPI and Cgroups v2.
"""
import os
import psutil
from datarobot_drum.drum.enum import ArgumentsOptions

class ResourceMonitor:
    # ... existing __init__ and other methods ...

    def _collect_memory_info_in_docker(self):
        """
        Collect memory info inside Docker, supporting both Cgroups v1 and v2.
        """
        # Cgroups v1 paths
        mem_v1_path = "/sys/fs/cgroup/memory/"
        # Cgroups v2 paths
        mem_v2_path = "/sys/fs/cgroup/"

        if os.path.exists(os.path.join(mem_v1_path, "memory.limit_in_bytes")):
            # Cgroups v1
            total_bytes = int(open(os.path.join(mem_v1_path, "memory.limit_in_bytes")).read())
            usage_bytes = int(open(os.path.join(mem_v1_path, "memory.usage_in_bytes")).read())
            max_usage_bytes = int(open(os.path.join(mem_v1_path, "memory.max_usage_in_bytes")).read())
        elif os.path.exists(os.path.join(mem_v2_path, "memory.max")):
            # Cgroups v2
            # memory.max can contain "max" if no limit is set
            total_str = open(os.path.join(mem_v2_path, "memory.max")).read().strip()
            if total_str == "max":
                total_bytes = psutil.virtual_memory().total
            else:
                total_bytes = int(total_str)
            
            usage_bytes = int(open(os.path.join(mem_v2_path, "memory.current")).read().strip())
            
            # memory.peak is available in newer kernels for max usage
            if os.path.exists(os.path.join(mem_v2_path, "memory.peak")):
                max_usage_bytes = int(open(os.path.join(mem_v2_path, "memory.peak")).read().strip())
            else:
                max_usage_bytes = usage_bytes
        else:
            # Fallback to psutil if cgroups not found or unsupported version
            virtual_mem = psutil.virtual_memory()
            return {
                "total_mb": ByteConv.from_bytes(virtual_mem.total).mbytes,
                "usage_mb": ByteConv.from_bytes(virtual_mem.total - virtual_mem.available).mbytes,
                "max_usage_mb": ByteConv.from_bytes(virtual_mem.total - virtual_mem.available).mbytes
            }

        return {
            "total_mb": ByteConv.from_bytes(total_bytes).mbytes,
            "usage_mb": ByteConv.from_bytes(usage_bytes).mbytes,
            "max_usage_mb": ByteConv.from_bytes(max_usage_bytes).mbytes
        }

    def collect_drum_info(self):
        # ... existing get_proc_data ...

        if self._drum_proc is None:
            if self._is_drum_process:
                self._drum_proc = self._current_proc
            else:
                # Uvicorn/Gunicorn worker might be several levels deep
                # Climb up until we find 'drum'
                curr = self._current_proc
                while curr:
                    try:
                        # Check cmdline or name for 'drum'
                        cmdline = curr.cmdline()
                        if any("datarobot_drum.drum.entry_point" in arg for arg in cmdline) or \
                           (len(cmdline) > 0 and cmdline[0].endswith("drum")):
                            self._drum_proc = curr
                            break
                    except (psutil.AccessDenied, psutil.NoSuchProcess):
                        break
                    curr = curr.parent()
                
                # Fallback to original logic if not found
                if self._drum_proc is None:
                    parents = self._current_proc.parents()
                    for p in parents:
                        if p.name() == ArgumentsOptions.MAIN_COMMAND:
                            self._drum_proc = p
                            break

        # ... rest of the method remains the same ...
```

## Key Differences

| Feature | Cgroups v1 | Cgroups v2 |
|---------|------------|------------|
| Path | `/sys/fs/cgroup/memory/` | `/sys/fs/cgroup/` |
| Limit File | `memory.limit_in_bytes` | `memory.max` |
| Usage File | `memory.usage_in_bytes` | `memory.current` |
| Max Usage File | `memory.max_usage_in_bytes` | `memory.peak` |

## Framework-Agnostic Metadata

To remove reliance on `flask.request`, any metadata needed for resource monitoring (like request IDs or user info) should be passed explicitly to `collect_resources_info` or extracted from a standard `RequestState` object stored in `contextvars`, which works for both Flask and FastAPI.

## Process Discovery

The updated `collect_drum_info` is more robust for Uvicorn:
- It climbs the process tree and checks `cmdline` for `datarobot_drum.drum.entry_point`.
- This handles cases where Uvicorn might have a different intermediate process name.

## Notes

- `ByteConv` class remains unchanged.
- `psutil` fallback is added if cgroups cannot be read.
- For Cgroups v2, if `memory.max` is "max", we fallback to total host memory.
