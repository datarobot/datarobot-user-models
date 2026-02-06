# Removal Plan: custom_model_runner/datarobot_drum/drum/resource_monitor.py

Update `ResourceMonitor` to support Uvicorn process hierarchy and Cgroups v2.

## Current State

The file has a comment referencing Flask:
```python
# case with Flask server, there is only one process - drum
```

And the process discovery logic is tailored to Gunicorn's process structure.

## Actions

### Phase 1: Add Uvicorn Support

1. **Update process discovery for Uvicorn**:
```python
def collect_drum_info(self):
    if self._drum_proc is None:
        if self._is_drum_process:
            self._drum_proc = self._current_proc
        else:
            # Uvicorn/Gunicorn worker might be several levels deep
            curr = self._current_proc
            while curr:
                try:
                    cmdline = curr.cmdline()
                    if any("datarobot_drum.drum.entry_point" in arg for arg in cmdline) or \
                       (len(cmdline) > 0 and cmdline[0].endswith("drum")):
                        self._drum_proc = curr
                        break
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    break
                curr = curr.parent()
```

2. **Add Cgroups v2 support**:
```python
def _collect_memory_info_in_docker(self):
    # Cgroups v1 paths
    mem_v1_path = "/sys/fs/cgroup/memory/"
    # Cgroups v2 paths
    mem_v2_path = "/sys/fs/cgroup/"

    if os.path.exists(os.path.join(mem_v1_path, "memory.limit_in_bytes")):
        # Cgroups v1 logic
        pass
    elif os.path.exists(os.path.join(mem_v2_path, "memory.max")):
        # Cgroups v2 logic
        pass
    else:
        # Fallback to psutil
        pass
```

### Phase 2: Update Comments

Replace Flask-specific comments with framework-agnostic descriptions:
```python
# For single-process server (development mode), there is only one process - drum
```

## Key Differences

| Feature | Cgroups v1 | Cgroups v2 |
|---------|------------|------------|
| Path | `/sys/fs/cgroup/memory/` | `/sys/fs/cgroup/` |
| Limit File | `memory.limit_in_bytes` | `memory.max` |
| Usage File | `memory.usage_in_bytes` | `memory.current` |
| Max Usage File | `memory.max_usage_in_bytes` | `memory.peak` |

## Notes

- Process discovery needs to handle Uvicorn's deeper process tree
- Cgroups v2 is now default on newer container runtimes
- Fallback to psutil if cgroups cannot be read
