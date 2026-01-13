# Plan: custom_model_runner/datarobot_drum/drum/lazy_loading/

Fix `asyncio.run()` issue in `LazyLoadingHandler` when running under FastAPI.

## Overview

The `LazyLoadingHandler` uses `asyncio.run()` to download files in parallel. This works fine in synchronous Flask/Gunicorn environments but fails in FastAPI/Uvicorn because an event loop is already running in the worker process.

## Proposed Implementation

### 1. Update `LazyLoadingHandler.download_lazy_loading_files()`

Modify `lazy_loading_handler.py` to check for a running event loop before calling `asyncio.run()`.

```python
    def download_lazy_loading_files(self):
        if not self.is_lazy_loading_available:
            return
        logger.info("Start downloading lazy loading files")
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We are in a running event loop (FastAPI/Uvicorn)
            # We must use the existing loop to run the coroutine.
            # Since this method is usually called during initialization,
            # we can use loop.run_until_complete() IF we are in a main thread
            # or simply schedule it if it can be async.
            # However, download MUST finish before model is loaded.
            
            # Use nest_asyncio if needed, OR better:
            # Change download_lazy_loading_files to be async if possible.
            # For now, a safe way is to use a separate thread if we're in a loop.
            
            import threading
            def run_in_new_loop():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(self._download_in_parallel())
                finally:
                    new_loop.close()

            thread = threading.Thread(target=run_in_new_loop)
            thread.start()
            thread.join()
        else:
            asyncio.run(self._download_in_parallel())
            
        logger.info("Lazy loading files have been downloaded")
```

### 2. Alternative: Async Initialization

In the future, we should consider making `predictor.configure()` async, but for now, the threading approach is a safe workaround for the `asyncio.run()` limitation.

## Key Changes

1.  **Event Loop Detection**: Added detection of the existing event loop.
2.  **Threaded Workaround**: Running the download in a separate thread with its own loop ensures that we don't conflict with the Uvicorn event loop while still waiting for the download to complete before proceeding with model loading.

## Notes

- This change is backward compatible with Flask/Gunicorn.
- Ensures that models using lazy loading (e.g., large artifacts in S3) continue to work seamlessly in FastAPI.
