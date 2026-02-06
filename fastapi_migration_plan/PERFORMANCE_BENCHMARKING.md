# Performance Benchmarking: FastAPI vs Flask

This document outlines the methodology for comparing the performance of the new FastAPI/Uvicorn server against the existing Flask/Gunicorn implementation in DRUM.

## Methodology

We will measure three key metrics under varying load:
1. **Latency (p50, p95, p99)**: The time taken to process a single prediction request.
2. **Throughput (RPS)**: The number of requests per second the server can handle before saturation.
3. **Resource Utilization**: CPU and Memory usage of the server process.

### Test Environment
- **Hardware**: Dedicated benchmarking machine or cloud instance (e.g., AWS c5.large).
- **Tool**: `locust` for load generation.
- **Model**: A standard Scikit-Learn regression model (`sklearn_regression`).
- **Data**: A CSV file with 100 rows of features.

### Scenarios
1. **Single Request**: Baseline latency for a single user.
2. **Concurrent Load**: 10, 50, 100 concurrent users.
3. **Stress Test**: Increase load until the error rate > 1%.

## Benchmarking Script (locustfile.py)

```python
import os
from locust import HttpUser, task, between

class DrumUser(HttpUser):
    wait_time = between(0.1, 0.5)
    
    @task
    def predict(self):
        # Path to test data
        test_data_path = "tests/testdata/juniors_3_year_stats_regression.csv"
        with open(test_data_path, "rb") as f:
            self.client.post(
                "/predict/",
                files={"X": ("test.csv", f, "text/csv")}
            )

    def on_start(self):
        # Optional: pre-check server info
        self.client.get("/info/")
```

## Running the Benchmarks

### 1. Start Flask/Gunicorn Server
```bash
export MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE=gunicorn
drum server --code-dir model_templates/python3_sklearn --address localhost:8080 --target-type regression
```

### 2. Run Locust for Flask
```bash
locust -f locustfile.py --host http://localhost:8080 --users 50 --spawn-rate 10 --run-time 5m --csv flask_bench
```

### 3. Start FastAPI/Uvicorn Server
```bash
export MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE=fastapi
drum server --code-dir model_templates/python3_sklearn --address localhost:8080 --target-type regression
```

### 4. Run Locust for FastAPI
```bash
locust -f locustfile.py --host http://localhost:8080 --users 50 --spawn-rate 10 --run-time 5m --csv fastapi_bench
```

## Expected Results

FastAPI/Uvicorn is expected to show:
- Lower p99 latency due to better handling of concurrent requests by the ASGI loop.
- Higher throughput (RPS) in I/O bound scenarios (e.g., directAccess proxying).
- Similar latency for CPU-bound prediction tasks (since the actual model execution is still synchronous and offloaded to a thread pool).

## Advanced Tuning

### Scenario 5: Thread Pool Tuning
The `DRUM_FASTAPI_EXECUTOR_WORKERS` parameter controls the number of threads in the pool used for synchronous model execution. Tuning this is critical for CPU-bound tasks to avoid excessive context switching or underutilization.

**Methodology:**
1. Use a CPU-heavy model (e.g., large Random Forest or XGBoost).
2. Run Locust with high concurrency (e.g., 100 users).
3. Test with `DRUM_FASTAPI_EXECUTOR_WORKERS` set to 1, 4, 8, 16, 32.
4. Measure:
   - Average and p99 latency.
   - CPU utilization and context switches (using `pidstat` or `vmstat`).
   - Throughput (RPS).
