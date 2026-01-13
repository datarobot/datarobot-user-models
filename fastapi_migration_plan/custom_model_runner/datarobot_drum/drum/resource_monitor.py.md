# Plan: custom_model_runner/datarobot_drum/drum/resource_monitor.py

Ensure `ResourceMonitor` is framework-agnostic and properly integrated with FastAPI.

## Overview

The `ResourceMonitor` class collects memory and CPU information. While the current implementation appears to be framework-agnostic, we must ensure it remains so and is used correctly within the FastAPI event loop.

## Proposed Changes

### 1. Framework-Agnostic Interface

Ensure no Flask or FastAPI specific imports are added to `resource_monitor.py`. All request-specific metadata (if needed in the future) should be passed as explicit arguments.

### 2. Integration with FastAPI

In the FastAPI implementation of `PredictionServer`, the `/stats/` endpoint should call `collect_resources_info()`. Since `psutil` calls can be blocking, it's recommended to run this in a thread pool if performance becomes an issue, although for a single call it might be acceptable.

```python
@router.get("/stats/")
async def stats():
    """Endpoint for resource and performance statistics."""
    # Run sync collection in executor to avoid blocking the event loop
    ret_dict = await self._run_sync_in_executor(
        self._resource_monitor.collect_resources_info
    )
    
    if self._stats_collector:
        self._stats_collector.round()
        ret_dict["time_info"] = {}
        for name in self._stats_collector.get_report_names():
            d = self._stats_collector.dict_report(name)
            ret_dict["time_info"][name] = d
        self._stats_collector.stats_reset()
    
    return ret_dict
```

### 3. Cleanup

Register the `ResourceMonitor` for any necessary cleanup in the `FastAPIWorkerCtx`.

## Metadata and Request Tracking

If we need to track resource usage per request (as hinted in `REMOVING_FLASK_STRATEGY.md`), we should:
1. Define a `RequestState` dataclass in a common utility module.
2. Update `ResourceMonitor` methods to accept an optional `RequestState` object.
3. In FastAPI, populate `RequestState` from the `request` object and pass it to the monitor.

## Notes:
- The current implementation of `ResourceMonitor` uses `psutil` and `/sys/fs/cgroup/` which are framework-independent.
- Moving to FastAPI allows for more granular resource monitoring if we utilize async background tasks.
