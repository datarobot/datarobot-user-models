# Import DRUM's WSGI application
import os
from datarobot_drum import RuntimeParameters

workers = 10
if RuntimeParameters.has("CUSTOM_MODEL_WORKERS"):
    temp_workers = int(RuntimeParameters.get("CUSTOM_MODEL_WORKERS"))
    if 0 < temp_workers < 200:
        workers = temp_workers

backlog = 2048
if RuntimeParameters.has("DRUM_WEBSERVER_BACKLOG"):
    temp_backlog = int(RuntimeParameters.get("DRUM_WEBSERVER_BACKLOG"))
    if 1 <= temp_backlog <= 10000:
        backlog = temp_backlog

timeout = 120
if RuntimeParameters.has("DRUM_CLIENT_REQUEST_TIMEOUT"):
    temp_timeout = int(RuntimeParameters.get("DRUM_CLIENT_REQUEST_TIMEOUT"))
    if 0 <= temp_timeout <= 3600:
        timeout = temp_timeout

max_requests = 0
if RuntimeParameters.has("DRUM_GUNICORN_MAX_REQUESTS"):
    temp_max_requests = int(RuntimeParameters.get("DRUM_GUNICORN_MAX_REQUESTS"))
    if 0 <= temp_max_requests <= 10000:
        max_requests = temp_max_requests

max_requests_jitter = 0
if RuntimeParameters.has("DRUM_GUNICORN_MAX_REQUESTS_JITTER"):
    temp_max_requests_jitter = int(RuntimeParameters.get("DRUM_GUNICORN_MAX_REQUESTS_JITTER"))
    if 1 <= temp_max_requests_jitter <= 10000:
        max_requests_jitter = temp_max_requests_jitter

worker_connections = 5
if RuntimeParameters.has("DRUM_WORKER_CONNECTIONS"):
    temp_worker_connections = int(RuntimeParameters.get("DRUM_WORKER_CONNECTIONS"))
    if 1 <= temp_worker_connections <= 10000:
        worker_connections = temp_worker_connections

worker_class = "gevent"
if RuntimeParameters.has("DRUM_GUNICORN_WORKER_CLASS"):
    temp_worker_class = str(RuntimeParameters.get("DRUM_GUNICORN_WORKER_CLASS")).lower()
    if temp_worker_class in {"sync", "gevent"}:
        worker_class = temp_worker_class

if RuntimeParameters.has("DRUM_GUNICORN_GRACEFUL_TIMEOUT"):
    temp_graceful_timeout = int(RuntimeParameters.get("DRUM_GUNICORN_GRACEFUL_TIMEOUT"))
    if 1 <= temp_graceful_timeout <= 3600:
        graceful_timeout = temp_graceful_timeout

if RuntimeParameters.has("DRUM_GUNICORN_KEEP_ALIVE"):
    temp_keepalive = int(RuntimeParameters.get("DRUM_GUNICORN_KEEP_ALIVE"))
    if 1 <= temp_keepalive <= 3600:
        keepalive = temp_keepalive

loglevel = "info"
if RuntimeParameters.has("DRUM_GUNICORN_LOG_LEVEL"):
    temp_loglevel = str(RuntimeParameters.get("DRUM_GUNICORN_LOG_LEVEL")).lower()
    if temp_loglevel in {"debug", "info", "warning", "error", "critical"}:
        loglevel = temp_loglevel

bind = os.environ.get("ADDRESS", "0.0.0.0:8080")
# loglevel = "info"
accesslog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'


def post_worker_init(worker):
    from datarobot_drum.drum.gunicorn.app import app, set_worker_ctx
    from datarobot_drum.drum.gunicorn.context import create_ctx
    import sys, shlex

    # Build synthetic argv for DRUM so argparse sees a proper subcommand
    argv = []

    # Allow passing extra DRUM args via env var, e.g.:
    #   export DRUM_GUNICORN_DRUM_ARGS="--sidecar --gpu-predictor=nim --logging-level=info"
    extra = os.environ.get("DRUM_GUNICORN_DRUM_ARGS")
    if extra:
        argv.extend(shlex.split(extra))

    sys.argv = argv

    # Force single worker resources inside each gunicorn worker
    os.environ["MAX_WORKERS"] = "1"
    if RuntimeParameters.has("CUSTOM_MODEL_WORKERS"):
        os.environ.pop("MLOPS_RUNTIME_PARAM_CUSTOM_MODEL_WORKERS", None)

    ctx = create_ctx(app)
    set_worker_ctx(ctx)
    ctx.start()


def worker_exit(worker, code):
    """
    Handles the shutdown and cleanup operations for a single Gunicorn worker.

    When terminating or restarting a single Gunicorn worker, this method ensures
    that the shutdown and cleanup operations are applied only to that specific worker.
    The following steps should be executed per worker:
      - predictor.terminate()
      - stats_collector.mark("end")
      - runtime.cm_runner.terminate()
      - runtime.trace_provider.shutdown()
      - runtime.metric_provider.shutdown()
      - runtime.log_provider.shutdown()
      - etc.

    This is necessary because the Gunicorn server is started from the command line
    (outside of DRUM), and worker-specific cleanup must be handled explicitly.

    Args:
        worker: The Gunicorn worker instance being terminated.
        code: The exit code for the worker.
    """
    from datarobot_drum.drum.gunicorn.app import get_worker_ctx

    ctx = get_worker_ctx()
    if ctx:
        try:
            ctx.stop()
        finally:
            ctx.cleanup()
