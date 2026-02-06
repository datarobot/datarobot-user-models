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

---

## Memory Leak Detection

The `MemoryLeakDetector` class provides automated detection of memory leaks by analyzing memory growth trends over time.

### Overview

Memory leaks in ML prediction services can be subtle - a few KB per request that accumulate over hours or days. This detector:

- Samples memory usage at regular intervals
- Uses linear regression to detect upward trends
- Alerts when growth rate exceeds configurable threshold
- Provides metrics for dashboards and alerting

### Proposed Implementation

```python
"""
Memory leak detection for FastAPI DRUM server.

Features:
- Statistical trend analysis using linear regression
- Configurable thresholds and sample windows
- Integration with resource monitoring
- Alert callbacks for external systems

Usage:
    from datarobot_drum.drum.resource_monitor import MemoryLeakDetector
    
    detector = MemoryLeakDetector(
        sample_window=100,
        growth_threshold_mb_per_1k_requests=50.0,
    )
    
    # Record sample after each request (or periodically)
    detector.record_sample(process.memory_info().rss)
    
    # Check for leak
    is_leaking, message = detector.check_for_leak()
    if is_leaking:
        logger.warning(message)
"""
import logging
import time
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable

from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


@dataclass
class MemorySample:
    """Single memory sample with metadata."""
    request_count: int
    rss_bytes: int
    timestamp: float
    
    @property
    def rss_mb(self) -> float:
        return self.rss_bytes / (1024 * 1024)


class MemoryLeakDetector:
    """
    Detect memory leaks using statistical trend analysis.
    
    The detector uses linear regression to estimate memory growth rate
    and alerts when growth exceeds the configured threshold.
    
    Algorithm:
    1. Collect memory samples (RSS) after each request or periodically
    2. Maintain a sliding window of samples
    3. Calculate linear regression slope (memory growth per request)
    4. Alert if slope exceeds threshold (MB per 1000 requests)
    
    Example:
        detector = MemoryLeakDetector(
            sample_window=100,
            growth_threshold_mb_per_1k_requests=50.0,
        )
        
        # In request middleware or after processing
        import psutil
        rss = psutil.Process().memory_info().rss
        detector.record_sample(rss)
        
        # Periodic check (e.g., every 100 requests)
        is_leaking, message = detector.check_for_leak()
    """
    
    def __init__(
        self,
        sample_window: int = 100,
        growth_threshold_mb_per_1k_requests: float = 50.0,
        min_samples_for_detection: int = 20,
        on_leak_detected: Optional[Callable[[str, float], None]] = None,
    ):
        """
        Initialize memory leak detector.
        
        Args:
            sample_window: Number of samples to keep for trend analysis
            growth_threshold_mb_per_1k_requests: Alert threshold (MB per 1000 requests)
            min_samples_for_detection: Minimum samples needed before checking
            on_leak_detected: Optional callback(message, growth_rate) when leak detected
        """
        self.sample_window = sample_window
        self.growth_threshold = growth_threshold_mb_per_1k_requests
        self.min_samples = min_samples_for_detection
        self.on_leak_detected = on_leak_detected
        
        self._samples: deque[MemorySample] = deque(maxlen=sample_window)
        self._request_count = 0
        self._last_check_time = 0.0
        self._last_alert_time = 0.0
        self._alert_cooldown = 300.0  # 5 minutes between alerts
        
        # Baseline (initial memory after startup)
        self._baseline_rss: Optional[int] = None
        
        # Statistics
        self._leak_detected_count = 0
        self._max_growth_rate_seen = 0.0
    
    def record_sample(self, rss_bytes: int):
        """
        Record a memory sample.
        
        Call this after each request or at regular intervals.
        
        Args:
            rss_bytes: Current RSS (Resident Set Size) in bytes
        """
        self._request_count += 1
        
        # Set baseline on first sample
        if self._baseline_rss is None:
            self._baseline_rss = rss_bytes
        
        sample = MemorySample(
            request_count=self._request_count,
            rss_bytes=rss_bytes,
            timestamp=time.time()
        )
        self._samples.append(sample)
    
    def check_for_leak(self) -> Tuple[bool, Optional[str]]:
        """
        Check if memory growth indicates a leak.
        
        Returns:
            Tuple of (is_leaking, message)
            - is_leaking: True if leak detected
            - message: Description of leak if detected, None otherwise
        """
        if len(self._samples) < self.min_samples:
            return False, None
        
        # Calculate linear regression slope
        slope = self._calculate_slope()
        
        # Convert to MB per 1000 requests
        growth_per_1k = (slope * 1000) / (1024 * 1024)
        
        # Track max growth rate
        self._max_growth_rate_seen = max(self._max_growth_rate_seen, growth_per_1k)
        
        if growth_per_1k > self.growth_threshold:
            message = (
                f"Potential memory leak detected: "
                f"Memory growing at {growth_per_1k:.1f} MB per 1000 requests "
                f"(threshold: {self.growth_threshold:.1f} MB). "
                f"Current RSS: {self._samples[-1].rss_mb:.1f} MB, "
                f"Baseline: {self._baseline_rss / (1024*1024):.1f} MB"
            )
            
            # Check alert cooldown
            now = time.time()
            if now - self._last_alert_time > self._alert_cooldown:
                self._leak_detected_count += 1
                self._last_alert_time = now
                
                # Trigger callback if set
                if self.on_leak_detected:
                    try:
                        self.on_leak_detected(message, growth_per_1k)
                    except Exception as e:
                        logger.warning("Leak detection callback failed: %s", e)
                
                logger.warning(message)
            
            return True, message
        
        return False, None
    
    def _calculate_slope(self) -> float:
        """
        Calculate linear regression slope of memory vs request count.
        
        Uses simple linear regression: slope = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
        
        Returns:
            Slope in bytes per request
        """
        if len(self._samples) < 2:
            return 0.0
        
        x = [s.request_count for s in self._samples]
        y = [s.rss_bytes for s in self._samples]
        
        n = len(x)
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def get_statistics(self) -> dict:
        """Get memory statistics for monitoring endpoints."""
        if not self._samples:
            return {
                "sample_count": 0,
                "request_count": self._request_count,
            }
        
        rss_values = [s.rss_mb for s in self._samples]
        
        return {
            "sample_count": len(self._samples),
            "request_count": self._request_count,
            "current_rss_mb": self._samples[-1].rss_mb,
            "baseline_rss_mb": self._baseline_rss / (1024 * 1024) if self._baseline_rss else None,
            "min_rss_mb": min(rss_values),
            "max_rss_mb": max(rss_values),
            "avg_rss_mb": statistics.mean(rss_values),
            "growth_rate_mb_per_1k_requests": (self._calculate_slope() * 1000) / (1024 * 1024),
            "max_growth_rate_seen_mb_per_1k": self._max_growth_rate_seen,
            "leak_alerts_count": self._leak_detected_count,
            "threshold_mb_per_1k": self.growth_threshold,
        }
    
    def reset(self):
        """Reset detector state (e.g., after model reload)."""
        self._samples.clear()
        self._request_count = 0
        self._baseline_rss = None
        self._leak_detected_count = 0
        self._max_growth_rate_seen = 0.0
        logger.info("Memory leak detector reset")


class MemoryMonitorMiddleware:
    """
    ASGI middleware for automatic memory monitoring.
    
    Samples memory after each request and periodically checks for leaks.
    
    Usage:
        detector = MemoryLeakDetector()
        app.add_middleware(MemoryMonitorMiddleware, detector=detector)
    """
    
    def __init__(
        self,
        app,
        detector: MemoryLeakDetector,
        sample_interval: int = 10,  # Sample every N requests
        check_interval: int = 100,  # Check for leak every N requests
    ):
        self.app = app
        self.detector = detector
        self.sample_interval = sample_interval
        self.check_interval = check_interval
        self._request_counter = 0
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        await self.app(scope, receive, send)
        
        self._request_counter += 1
        
        # Sample memory periodically
        if self._request_counter % self.sample_interval == 0:
            import psutil
            rss = psutil.Process().memory_info().rss
            self.detector.record_sample(rss)
        
        # Check for leak periodically
        if self._request_counter % self.check_interval == 0:
            self.detector.check_for_leak()
```

### Integration with ResourceMonitor

Add to the existing `ResourceMonitor` class:

```python
class ResourceMonitor:
    def __init__(self, ...):
        # ... existing init ...
        
        # Memory leak detection
        self.leak_detector = MemoryLeakDetector(
            sample_window=100,
            growth_threshold_mb_per_1k_requests=50.0,
        )
    
    def collect_resources_info(self, ...):
        # ... existing collection ...
        
        # Record memory sample
        self.leak_detector.record_sample(rss_bytes)
        
        # Add leak detection status
        result["memory_leak_stats"] = self.leak_detector.get_statistics()
        
        return result
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DRUM_MEMORY_LEAK_THRESHOLD_MB` | Growth threshold (MB/1k requests) | `50.0` |
| `DRUM_MEMORY_LEAK_SAMPLE_WINDOW` | Number of samples to analyze | `100` |
| `DRUM_MEMORY_LEAK_ENABLED` | Enable leak detection | `true` |

### Testing

```python
# tests/unit/datarobot_drum/drum/test_memory_leak_detector.py
import pytest
from datarobot_drum.drum.resource_monitor import MemoryLeakDetector

class TestMemoryLeakDetector:
    def test_detects_linear_leak(self):
        detector = MemoryLeakDetector(
            sample_window=50,
            growth_threshold_mb_per_1k_requests=10.0,
            min_samples_for_detection=10,
        )
        
        # Simulate linear memory growth (100 KB per request = 100 MB per 1k)
        base_memory = 100 * 1024 * 1024  # 100 MB
        for i in range(50):
            detector.record_sample(base_memory + i * 100 * 1024)
        
        is_leaking, message = detector.check_for_leak()
        assert is_leaking
        assert "memory leak" in message.lower()
    
    def test_no_false_positive_stable_memory(self):
        detector = MemoryLeakDetector(
            sample_window=50,
            growth_threshold_mb_per_1k_requests=10.0,
            min_samples_for_detection=10,
        )
        
        # Stable memory with small random fluctuations
        import random
        base_memory = 100 * 1024 * 1024
        for i in range(50):
            noise = random.randint(-1024 * 100, 1024 * 100)
            detector.record_sample(base_memory + noise)
        
        is_leaking, _ = detector.check_for_leak()
        assert not is_leaking
    
    def test_statistics(self):
        detector = MemoryLeakDetector()
        
        for i in range(30):
            detector.record_sample(100 * 1024 * 1024 + i * 1024)
        
        stats = detector.get_statistics()
        assert stats["sample_count"] == 30
        assert "current_rss_mb" in stats
        assert "growth_rate_mb_per_1k_requests" in stats
```
