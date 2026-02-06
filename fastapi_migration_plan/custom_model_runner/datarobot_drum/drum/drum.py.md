# Plan: custom_model_runner/datarobot_drum/drum/drum.py

Generalize `CMRunner` to support both Flask and FastAPI app instances.

## Changes:

### 1. Update `CMRunner.__init__`
```python
class CMRunner:
    def __init__(self, runtime, app=None, worker_ctx=None):
        self.runtime = runtime
        self.app = app  # Renamed from flask_app, can be Flask or FastAPI
        self.worker_ctx = worker_ctx # Can be WorkerCtx (Gunicorn) or FastAPIWorkerCtx (Uvicorn)
        # ... rest unchanged ...
```

### 2. Update `CMRunner._run_predictions`
```python
    def _run_predictions(self, stats_collector: Optional[StatsCollector] = None):
        # ... (keep validation and logging) ...

        params = self.get_predictor_params()
        predictor = None
        try:
            from datarobot_drum.drum.root_predictors.prediction_server import PredictionServer

            if stats_collector:
                stats_collector.mark("start")
            
            # PredictionServer now handles both app types
            predictor = (
                PredictionServer(params, app=self.app, worker_ctx=self.worker_ctx)
                if self.run_mode == RunMode.SERVER
                else GenericPredictorComponent(params)
            )
            # ... rest unchanged ...
```

## Key Considerations:
- Parameter `flask_app` is renamed to `app` to be framework-agnostic.
- The `worker_ctx` remains generic as both implementations will follow the same protocol for `defer_cleanup`.
- Existing Flask-only code will continue to work as long as the interface is maintained.
