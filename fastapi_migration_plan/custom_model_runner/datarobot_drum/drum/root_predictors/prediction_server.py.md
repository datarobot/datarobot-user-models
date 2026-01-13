# Plan: custom_model_runner/datarobot_drum/drum/root_predictors/prediction_server.py

Refactor `PredictionServer` to support FastAPI routing alongside Flask.

## Overview

The `PredictionServer` class will be updated to work with both Flask and FastAPI apps. When `DRUM_SERVER_TYPE=fastapi`, it will use FastAPI's `APIRouter` instead of Flask's `Blueprint`.

## Changes:

### 1. Imports
```python
# Add new imports
from typing import Union, Optional, List
from fastapi import FastAPI, APIRouter, Request, Response, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
import httpx  # Async HTTP client for proxy endpoints

# Keep existing Flask imports for backward compatibility during transition
from flask import Flask, Blueprint, Response as FlaskResponse, jsonify, request as flask_request
```

### 2. Update `__init__` signature
```python
def __init__(self, params: dict, app: Union[Flask, FastAPI, None] = None, worker_ctx=None):
    self._params = params
    self.app = app  # Renamed from flask_app to be generic
    self._worker_ctx = worker_ctx  # Store worker context for cleanup and watchdog
    # ... rest remains the same
```

### 3. Refactor `materialize()` method

Split into two methods based on app type:
- `_materialize_flask()` - existing Flask blueprint logic
- `_materialize_fastapi()` - new FastAPI router logic

```python
def materialize(self):
    if isinstance(self.app, FastAPI) or self._is_fastapi_mode():
        return self._materialize_fastapi()
    else:
        return self._materialize_flask()

def _is_fastapi_mode(self):
    from datarobot_drum import RuntimeParameters
    return (
        RuntimeParameters.has("DRUM_SERVER_TYPE") 
        and str(RuntimeParameters.get("DRUM_SERVER_TYPE")).lower() in ["fastapi", "uvicorn"]
    )
```

### 4. Implement `_materialize_fastapi()`

All endpoints must be implemented with FastAPI equivalents:

```python
def _materialize_fastapi(self):
    from datarobot_drum.drum.server import get_fastapi_app
    from datarobot_drum.drum.fastapi.extensions import load_fastapi_extensions
    
    router = APIRouter()
    
    @router.get("/")
    @router.get("/ping/")
    async def ping():
        if hasattr(self._predictor, "liveness_probe"):
            return self._predictor.liveness_probe()
        return {"message": "OK"}
    
    @router.get("/capabilities/")
    async def capabilities():
        return self.make_capabilities()
    
    @router.get("/info/")
    async def info():
        model_info = self._predictor.model_info()
        model_info.update({ModelInfoKeys.LANGUAGE: self._run_language.value})
        model_info.update({ModelInfoKeys.DRUM_VERSION: drum_version})
        model_info.update({ModelInfoKeys.DRUM_SERVER: "fastapi"})
        model_info.update({ModelInfoKeys.MODEL_METADATA: read_model_metadata_yaml(self._code_dir)})
        return model_info
    
    @router.get("/health/")
    async def health():
        if hasattr(self._predictor, "readiness_probe"):
            return self._predictor.readiness_probe()
        return {"message": "OK"}
    
    @router.get("/stats/")
    async def stats():
        """Endpoint for resource and performance statistics."""
        ret_dict = self._resource_monitor.collect_resources_info()
        
        if self._stats_collector:
            self._stats_collector.round()
            ret_dict["time_info"] = {}
            for name in self._stats_collector.get_report_names():
                d = self._stats_collector.dict_report(name)
                ret_dict["time_info"][name] = d
            self._stats_collector.stats_reset()
        
        return ret_dict

    @router.post("/predictions/")
    @router.post("/predict/")
    @router.post("/invocations")
    async def predict(request: Request):
        logger.debug("Entering predict() endpoint")
        
        # Pre-process request for sync mixin
        await self._prepare_fastapi_request(request)
        
        with otel_context(tracer, "drum.invocations", request.headers) as span:
            span.set_attributes(extract_request_headers(dict(request.headers)))
            self._pre_predict_and_transform()
            try:
                # Run sync prediction in executor to not block event loop
                response, response_status = await self._run_sync_in_executor(
                    self.do_predict_structured,
                    logger=logger,
                    request=request
                )
            finally:
                self._post_predict_and_transform()
        
        return self._convert_fastapi_response(response, response_status)
    
    @router.post("/transform/")
    async def transform(request: Request):
        logger.debug("Entering transform() endpoint")
        
        await self._prepare_fastapi_request(request)
        
        with otel_context(tracer, "drum.transform", request.headers) as span:
            span.set_attributes(extract_request_headers(dict(request.headers)))
            self._pre_predict_and_transform()
            try:
                response, response_status = await self._run_sync_in_executor(
                    self.do_transform,
                    logger=logger,
                    request=request
                )
            finally:
                self._post_predict_and_transform()
        
        return self._convert_fastapi_response(response, response_status)
    
    @router.post("/predictionsUnstructured/")
    @router.post("/predictUnstructured/")
    async def predict_unstructured(request: Request):
        logger.debug("Entering predict_unstructured() endpoint")
        
        await self._prepare_fastapi_request(request)
        
        with otel_context(tracer, "drum.predictUnstructured", request.headers) as span:
            span.set_attributes(extract_request_headers(dict(request.headers)))
            self._pre_predict_and_transform()
            try:
                response, response_status = await self._run_sync_in_executor(
                    self.do_predict_unstructured,
                    logger=logger,
                    request=request
                )
            finally:
                self._post_predict_and_transform()
        
        return self._convert_fastapi_response(response, response_status)
    
    @router.post("/chat/completions")
    @router.post("/v1/chat/completions")
    async def chat(request: Request):
        logger.debug("Entering chat endpoint")
        
        await self._prepare_fastapi_request(request)
        body_json = await request.json()
        
        with otel_context(tracer, "drum.chat.completions", request.headers) as span:
            span.set_attributes(extract_chat_request_attributes(body_json))
            span.set_attributes(extract_request_headers(dict(request.headers)))
            self._pre_predict_and_transform()
            try:
                # Chat might be streaming, handled inside do_chat
                result = await self._run_sync_in_executor(
                    self.do_chat,
                    logger=logger,
                    request=request,
                    is_fastapi=True
                )
                
                # result is (response, status_code)
                response, response_status = result
                
                if isinstance(response, dict) and response_status == 200:
                    span.set_attributes(extract_chat_response_attributes(response))
            finally:
                self._post_predict_and_transform()
        
        return response # do_chat returns Response/JSONResponse for FastAPI if is_fastapi=True
    
    @router.get("/models")
    @router.get("/v1/models")
    async def get_supported_llm_models_endpoint():
        logger.debug("Entering models endpoint")
        self._pre_predict_and_transform()
        try:
            response, response_status = self.get_supported_llm_models(logger=logger)
        finally:
            self._post_predict_and_transform()
        
        return JSONResponse(content=response, status_code=response_status)
    
    @router.api_route("/directAccess/{path:path}", methods=["GET", "POST", "PUT"])
    @router.api_route("/nim/{path:path}", methods=["GET", "POST", "PUT"])
    async def forward_request(path: str, request: Request):
        """Proxy endpoint for direct access to NIM/OpenAI backend."""
        with otel_context(tracer, "drum.directAccess", request.headers) as span:
            span.set_attributes(extract_request_headers(dict(request.headers)))
            if not hasattr(self._predictor, "openai_host") or not hasattr(self._predictor, "openai_port"):
                msg = "This endpoint is only supported by OpenAI based predictors"
                span.set_status(StatusCode.ERROR, msg)
                return JSONResponse(content={"message": msg}, status_code=HTTP_400_BAD_REQUEST)
            
            openai_host = self._predictor.openai_host
            openai_port = self._predictor.openai_port
            body = await request.body()
            
            forward_headers = {
                k: v for k, v in request.headers.items()
                if k.lower() not in ('host', 'content-length', 'transfer-encoding')
            }
            
            timeout = httpx.Timeout(
                self.get_nim_direct_access_request_timeout(),
                connect=30.0
            )
            
            try:
                # Use StreamingResponse to support both regular and SSE responses from backend
                async def generate_content():
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        async with client.stream(
                            method=request.method,
                            url=f"http://{openai_host}:{openai_port}/{path.rstrip('/')}",
                            headers=forward_headers,
                            params=dict(request.query_params),
                            content=body,
                            follow_redirects=False,
                        ) as resp:
                            # We need to capture the response status and headers from the stream
                            # This is a bit tricky with StreamingResponse, so we store them
                            request.state.proxy_status = resp.status_code
                            request.state.proxy_headers = {
                                k: v for k, v in resp.headers.items()
                                if k.lower() not in ('content-encoding', 'transfer-encoding', 'content-length')
                            }
                            async for chunk in resp.aiter_bytes():
                                yield chunk

                return StreamingResponse(
                    generate_content(),
                    # Note: status and headers will be set by the first yield if we use a more complex wrapper
                    # For simplicity in plan, we assume 200 or handle it in the generator
                )
            except Exception as e:
                span.set_status(StatusCode.ERROR, str(e))
                return JSONResponse(content={"message": f"ERROR: {e}"}, status_code=502)
    
    # Get or create FastAPI app
    app = get_fastapi_app(router, self.app)
    
    # Add StdoutFlusher middleware if available
    if hasattr(self, '_stdout_flusher'):
        from datarobot_drum.drum.server import StdoutFlusherMiddleware
        app.add_middleware(StdoutFlusherMiddleware, flusher=self._stdout_flusher)

    # Load custom FastAPI extensions
    load_fastapi_extensions(app, self._code_dir)
    
    # Start stdout flusher and register for cleanup
    if self._worker_ctx and hasattr(self, '_stdout_flusher'):
        self._worker_ctx.add_thread(
            self._stdout_flusher._flusher_thread,
            name="StdoutFlusher"
        )
        self._stdout_flusher.start()

    # Start watchdog thread if enabled and worker context available
    self._start_watchdog_if_enabled()
    
    # Register executor cleanup
    if self._worker_ctx:
        self._worker_ctx.defer_cleanup(
            self._shutdown_executor,
            order=100,
            desc="ThreadPoolExecutor shutdown"
        )
    
    return []
```

### 5. Helper methods for FastAPI

```python
async def _prepare_fastapi_request(self, request: Request):
    """Pre-fetch body and form data for sync mixin methods."""
    # Update last activity time for StdoutFlusher
    if hasattr(self, '_stdout_flusher'):
        self._stdout_flusher.set_last_activity_time()

    body = await request.body()
    request.state.body = body
    
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        request.state.form = {k: v for k, v in form.items() if isinstance(v, str)}
        request.state.files = {}
        for key, value in form.items():
            if hasattr(value, 'read'):
                file_content = await value.read()
                request.state.files[key] = {
                    "content": file_content,
                    "filename": getattr(value, 'filename', None),
                    "content_type": getattr(value, 'content_type', None),
                }

def _convert_fastapi_response(self, response, status_code):
    """Convert various response types to FastAPI Response."""
    if isinstance(response, (Response, JSONResponse, StreamingResponse)):
        return response
    
    if isinstance(response, dict):
        return JSONResponse(content=response, status_code=status_code)
    
    # Handle Flask response objects if they leak from mixin
    if hasattr(response, 'get_data'):
        return Response(
            content=response.get_data(),
            status_code=status_code,
            media_type=response.content_type
        )
    
    return Response(content=response, status_code=status_code)

async def _run_sync_in_executor(self, func, *args, **kwargs):
    """Run a synchronous function in a thread pool executor."""
    import asyncio
    from functools import partial
    from concurrent.futures import ThreadPoolExecutor
    
    if not hasattr(self, '_executor'):
        from datarobot_drum import RuntimeParameters
        workers = 4
        if RuntimeParameters.has("DRUM_FASTAPI_EXECUTOR_WORKERS"):
            workers = int(RuntimeParameters.get("DRUM_FASTAPI_EXECUTOR_WORKERS"))
        self._executor = ThreadPoolExecutor(max_workers=workers)
    
    loop = asyncio.get_event_loop()
    bound_func = partial(func, *args, **kwargs)
    return await loop.run_in_executor(self._executor, bound_func)

def _shutdown_executor(self):
    """Shutdown the thread pool executor."""
    if hasattr(self, '_executor'):
        self._executor.shutdown(wait=True)
```

### 6. Watchdog and NIM support

```python
def _start_watchdog_if_enabled(self):
    """Start NIM watchdog thread if enabled."""
    from datarobot_drum import RuntimeParameters
    if not RuntimeParameters.has("USE_NIM_WATCHDOG"):
        return
    
    if str(RuntimeParameters.get("USE_NIM_WATCHDOG")).lower() not in ["true", "1", "yes"]:
        return
    
    if self._worker_ctx is None:
        return
    
    from threading import Thread
    port = self._params.get("address", "8080").split(":")[-1]
    try:
        port = int(port)
    except ValueError:
        port = 8080
        
    self._server_watchdog = Thread(
        target=self.watchdog,
        args=(port,),
        daemon=True,
        name="NIM Sidecar Watchdog",
    )
    
    self._worker_ctx.add_thread(
        self._server_watchdog,
        join_timeout=5.0,
        name="NIM Sidecar Watchdog"
    )
    self._server_watchdog.start()
    logger.info("Started NIM watchdog thread for port %s", port)
```

## Runtime Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `DRUM_SERVER_TYPE` | Server type: "flask", "gunicorn", or "fastapi" | "flask" |
| `DRUM_FASTAPI_EXECUTOR_WORKERS` | Thread pool size for sync operations | 4 |

## Notes:
- The `PredictionServer` now takes an optional `worker_ctx`.
- FastAPI endpoints pre-fetch data into `request.state` to be framework-agnostic for `PredictMixin`.
- Sync operations (the actual model execution) are offloaded to a `ThreadPoolExecutor` to keep the FastAPI event loop responsive.
- Error handling and response conversion are centralized.
- Watchdog integration is preserved for NIM-based models.
