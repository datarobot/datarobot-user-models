# Plan: custom_model_runner/datarobot_drum/drum/language_predictors/java_predictor/java_predictor.py

Replace `werkzeug` dependency with framework-agnostic logic and ensure Java predictor works with FastAPI.

## Overview

The `JavaPredictor` class has special integration requirements:
1. Uses `werkzeug.datastructures.ImmutableMultiDict` for query parameters
2. Communicates with Java process via subprocess/socket
3. Has specific timeout and resource management needs

## Migration Checklist

- [ ] Replace Werkzeug `ImmutableMultiDict` with framework-agnostic solution
- [ ] Ensure Java subprocess management works with Uvicorn process model
- [ ] Test timeout behavior with async request handling
- [ ] Verify resource cleanup on worker shutdown

## Changes:

### 1. Framework-Agnostic Query Parameter Handling

```python
"""
Java predictor module - framework-agnostic version.

Supports both Flask/Werkzeug and FastAPI/Starlette request handling.
"""
import logging
from typing import Any, Dict, Mapping, Union

from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

# Framework-agnostic MultiDict handling
_MULTIDICT_TYPES = []

try:
    from werkzeug.datastructures import ImmutableMultiDict as WerkzeugMultiDict
    _MULTIDICT_TYPES.append(WerkzeugMultiDict)
except ImportError:
    WerkzeugMultiDict = None

try:
    from starlette.datastructures import ImmutableMultiDict as StarletteMultiDict
    _MULTIDICT_TYPES.append(StarletteMultiDict)
except ImportError:
    StarletteMultiDict = None

try:
    from starlette.datastructures import QueryParams
    _MULTIDICT_TYPES.append(QueryParams)
except ImportError:
    QueryParams = None


def _normalize_query_params(query: Any) -> Dict[str, str]:
    """
    Normalize query parameters from any framework to a simple dict.
    
    Handles:
    - Plain dict
    - Werkzeug ImmutableMultiDict
    - Starlette ImmutableMultiDict / QueryParams
    - Any Mapping type
    
    Args:
        query: Query parameters in various formats
        
    Returns:
        Simple dict with string keys and values
    """
    if query is None:
        return {}
    
    if isinstance(query, dict):
        return query
    
    # Check for MultiDict types (Werkzeug or Starlette)
    for multidict_type in _MULTIDICT_TYPES:
        if multidict_type is not None and isinstance(query, multidict_type):
            # MultiDict.to_dict() returns first value for each key
            if hasattr(query, "to_dict"):
                return query.to_dict()
            # QueryParams doesn't have to_dict, but is dict-like
            return dict(query)
    
    # Fallback for any Mapping type
    if isinstance(query, Mapping):
        return dict(query)
    
    logger.warning(
        "Unexpected query parameter type: %s. Attempting dict conversion.",
        type(query).__name__
    )
    try:
        return dict(query)
    except (TypeError, ValueError) as e:
        logger.error("Failed to convert query parameters: %s", e)
        return {}
```

### 2. Update `predict_unstructured` method

```python
def predict_unstructured(self, data, **kwargs):
    """
    Handle unstructured prediction for Java models.
    
    Framework-agnostic implementation that works with both
    Flask/Werkzeug and FastAPI/Starlette.
    """
    from datarobot_drum.drum.common import UnstructuredDtoKeys
    
    # Normalize query parameters (framework-agnostic)
    query = kwargs.get(UnstructuredDtoKeys.QUERY, {})
    query_dict = _normalize_query_params(query)
    
    # Get other parameters
    mimetype = kwargs.get(UnstructuredDtoKeys.MIMETYPE, "application/octet-stream")
    charset = kwargs.get(UnstructuredDtoKeys.CHARSET, "utf-8")
    
    # Call Java predictor with normalized parameters
    result = self._call_java_predictor(
        data=data,
        query=query_dict,
        mimetype=mimetype,
        charset=charset
    )
    
    return result
```

### 3. Java Subprocess Management with Uvicorn

```python
class JavaPredictor:
    """
    Java model predictor.
    
    Important considerations for FastAPI/Uvicorn:
    - Uvicorn uses multiple worker processes (not threads like Gunicorn+gevent)
    - Each worker needs its own Java subprocess
    - Subprocess must be started after fork, not before
    - Graceful shutdown requires proper subprocess cleanup
    """
    
    def __init__(self):
        self._java_process = None
        self._initialized = False
        self._lock = threading.Lock()
    
    def _ensure_initialized(self):
        """
        Lazy initialization of Java subprocess.
        
        IMPORTANT: Must be called after Uvicorn worker fork,
        not during module import or app creation.
        """
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            logger.info("Initializing Java predictor subprocess...")
            self._start_java_process()
            self._initialized = True
    
    def _start_java_process(self):
        """Start the Java predictor subprocess."""
        import subprocess
        import os
        
        java_home = os.environ.get("JAVA_HOME", "")
        java_cmd = os.path.join(java_home, "bin", "java") if java_home else "java"
        
        # Start Java process
        self._java_process = subprocess.Popen(
            [java_cmd, "-jar", self._jar_path, "--port", str(self._port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Important: create new process group for clean shutdown
            start_new_session=True
        )
        
        # Wait for Java process to be ready
        self._wait_for_java_ready()
    
    def shutdown(self):
        """
        Gracefully shutdown Java subprocess.
        
        Called by FastAPIWorkerCtx during worker shutdown.
        """
        if self._java_process:
            logger.info("Shutting down Java predictor subprocess...")
            
            # Try graceful shutdown first
            self._java_process.terminate()
            
            try:
                self._java_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Java process did not terminate gracefully, killing...")
                self._java_process.kill()
                self._java_process.wait()
            
            self._java_process = None
            self._initialized = False
            logger.info("Java predictor subprocess stopped")
```

### 4. Integration with FastAPI Worker Context

```python
# In fastapi/context.py

class FastAPIWorkerCtx:
    """Worker context for FastAPI/Uvicorn."""
    
    def _setup_java_predictor(self, predictor):
        """
        Setup Java predictor for this worker.
        
        Java predictors need special handling:
        - Subprocess must be started after fork
        - Must be cleaned up on worker shutdown
        """
        if hasattr(predictor, "_ensure_initialized"):
            predictor._ensure_initialized()
        
        if hasattr(predictor, "shutdown"):
            self.defer_cleanup(
                predictor.shutdown,
                order=50,  # Before executor shutdown
                desc="Java predictor subprocess"
            )
```

### 5. Timeout Handling for Java Calls

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class JavaPredictor:
    # ... other methods ...
    
    async def predict_async(self, data, timeout: float = 120.0, **kwargs):
        """
        Async wrapper for Java prediction with timeout.
        
        Runs the sync Java call in a thread pool with timeout protection.
        """
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="java-predict-")
        
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    lambda: self.predict_unstructured(data, **kwargs)
                ),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            logger.error("Java prediction timed out after %.1fs", timeout)
            raise
        finally:
            executor.shutdown(wait=False)
```

## Testing

```python
# tests/unit/datarobot_drum/drum/language_predictors/test_java_predictor.py

import pytest
from unittest.mock import MagicMock, patch

class TestJavaPredictorQueryParams:
    """Test framework-agnostic query parameter handling."""
    
    def test_plain_dict(self):
        query = {"key": "value", "num": "123"}
        result = _normalize_query_params(query)
        assert result == {"key": "value", "num": "123"}
    
    def test_werkzeug_multidict(self):
        pytest.importorskip("werkzeug")
        from werkzeug.datastructures import ImmutableMultiDict
        
        query = ImmutableMultiDict([("key", "value1"), ("key", "value2")])
        result = _normalize_query_params(query)
        # to_dict() returns first value
        assert result == {"key": "value1"}
    
    def test_starlette_query_params(self):
        pytest.importorskip("starlette")
        from starlette.datastructures import QueryParams
        
        query = QueryParams("key=value&num=123")
        result = _normalize_query_params(query)
        assert result == {"key": "value", "num": "123"}
    
    def test_none_input(self):
        assert _normalize_query_params(None) == {}
    
    def test_unknown_type_fallback(self):
        # Should attempt dict() conversion
        class CustomMapping:
            def __iter__(self):
                return iter([("key", "value")])
            def keys(self):
                return ["key"]
            def __getitem__(self, key):
                return "value"
        
        result = _normalize_query_params(CustomMapping())
        assert result == {"key": "value"}
```

## Notes

- **Process Model:** Uvicorn forks workers, so Java subprocess must be started after fork
- **Resource Cleanup:** Use `defer_cleanup()` in worker context for proper shutdown
- **Timeout:** Wrap sync Java calls with asyncio timeout for consistent behavior
- **Framework Agnostic:** Query param handling works with both Flask and FastAPI
