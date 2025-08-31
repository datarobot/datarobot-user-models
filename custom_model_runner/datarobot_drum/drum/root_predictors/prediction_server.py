"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import os
import sys
import threading
import time
from pathlib import Path
from threading import Thread
import subprocess
import signal

import psutil
import requests
from flask import Response, jsonify, request
from werkzeug.exceptions import HTTPException
from werkzeug.serving import WSGIRequestHandler

from opentelemetry import trace

from datarobot_drum import RuntimeParameters
from datarobot_drum.drum.description import version as drum_version
from datarobot_drum.drum.enum import (
    FLASK_EXT_FILE_NAME,
    GPU_PREDICTORS,
    LOGGER_NAME_PREFIX,
    TARGET_TYPE_ARG_KEYWORD,
    ModelInfoKeys,
    RunLanguage,
    TargetType,
    URL_PREFIX_ENV_VAR_NAME,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.model_metadata import read_model_metadata_yaml
from datarobot_drum.drum.resource_monitor import ResourceMonitor
from datarobot_drum.drum.root_predictors.deployment_config_helpers import (
    parse_validate_deployment_config_file,
)
from datarobot_drum.drum.root_predictors.predict_mixin import PredictMixin
from datarobot_drum.drum.root_predictors.stdout_flusher import StdoutFlusher
from datarobot_drum.drum.server import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_500_INTERNAL_SERVER_ERROR,
    base_api_blueprint,
    get_flask_app,
)
from datarobot_drum.profiler.stats_collector import StatsCollector, StatsOperation
from datarobot_drum.drum.common import (
    otel_context,
    extract_chat_request_attributes,
    extract_chat_response_attributes,
)
from opentelemetry.trace.status import StatusCode

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


tracer = trace.get_tracer(__name__)


class TimeoutWSGIRequestHandler(WSGIRequestHandler):
    timeout = 3600
    if RuntimeParameters.has("DRUM_CLIENT_REQUEST_TIMEOUT"):
        timeout = int(RuntimeParameters.get("DRUM_CLIENT_REQUEST_TIMEOUT"))


class PredictionServer(PredictMixin):
    def __init__(self, params: dict):
        self._params = params
        self._show_perf = self._params.get("show_perf")
        self._resource_monitor = ResourceMonitor(monitor_current_process=True)
        self._run_language = RunLanguage(params.get("run_language"))
        self._gpu_predictor_type = self._params.get("gpu_predictor")
        self._target_type = TargetType(params[TARGET_TYPE_ARG_KEYWORD])
        self._code_dir = self._params.get("__custom_model_path__")
        self._deployment_config = parse_validate_deployment_config_file(
            self._params["deployment_config"]
        )
        self._stdout_flusher = StdoutFlusher()

        self._stats_collector = StatsCollector(disable_instance=not self._show_perf)
        self._stats_collector.register_report(
            "run_predictor_total", "finish", StatsOperation.SUB, "start"
        )
        self._predictor = self._setup_predictor()
        self._server_watchdog = None

    def _setup_predictor(self):
        if self._run_language == RunLanguage.PYTHON:
            from datarobot_drum.drum.language_predictors.python_predictor.python_predictor import (
                PythonPredictor,
            )

            predictor = PythonPredictor()
        elif self._run_language == RunLanguage.JAVA:
            from datarobot_drum.drum.language_predictors.java_predictor.java_predictor import (
                JavaPredictor,
            )

            predictor = JavaPredictor()
        elif self._run_language == RunLanguage.JULIA:
            from datarobot_drum.drum.language_predictors.julia_predictor.julia_predictor import (
                JlPredictor,
            )

            predictor = JlPredictor()
        elif self._run_language == RunLanguage.R:
            # this import is here, because RPredictor imports rpy library,
            # which is not installed for Java and Python cases.
            from datarobot_drum.drum.language_predictors.r_predictor.r_predictor import (
                RPredictor,
            )

            predictor = RPredictor()
        elif self._gpu_predictor_type and self._gpu_predictor_type == GPU_PREDICTORS.TRITON:
            from datarobot_drum.drum.gpu_predictors.triton_predictor import (
                TritonPredictor,
            )

            predictor = TritonPredictor()
        elif self._gpu_predictor_type and self._gpu_predictor_type == GPU_PREDICTORS.NIM:
            from datarobot_drum.drum.gpu_predictors.nim_predictor import NIMPredictor

            predictor = NIMPredictor()
        elif self._gpu_predictor_type and self._gpu_predictor_type == GPU_PREDICTORS.VLLM:
            from datarobot_drum.drum.gpu_predictors.vllm_predictor import VllmPredictor

            predictor = VllmPredictor()
        else:
            raise DrumCommonException(
                "Prediction server doesn't support language: {} ".format(self._run_language)
            )

        self._stdout_flusher.start()
        predictor.configure(self._params)
        return predictor

    def _terminate(self):
        if hasattr(self._predictor, "terminate"):
            self._predictor.terminate()
        self._stdout_flusher.stop()

    def _pre_predict_and_transform(self):
        self._stats_collector.enable()
        self._stats_collector.mark("start")

    def _post_predict_and_transform(self):
        self._stats_collector.mark("finish")
        self._stats_collector.disable()
        self._stdout_flusher.set_last_activity_time()

    @staticmethod
    def get_nim_direct_access_request_timeout():
        """
        Returns the timeout value for NIM direct access requests.
        Checks the 'NIM_DIRECT_ACCESS_REQUEST_TIMEOUT' runtime parameter; if not set, defaults to 3600 seconds.
        """
        timeout = 3600
        if RuntimeParameters.has("NIM_DIRECT_ACCESS_REQUEST_TIMEOUT"):
            timeout = int(RuntimeParameters.get("NIM_DIRECT_ACCESS_REQUEST_TIMEOUT"))
        return timeout

    def materialize(self):
        model_api = base_api_blueprint(self._terminate, self._predictor)

        @model_api.route("/capabilities/", methods=["GET"])
        def capabilities():
            return self.make_capabilities()

        @model_api.route("/info/", methods=["GET"])
        def info():
            model_info = self._predictor.model_info()
            model_info.update({ModelInfoKeys.LANGUAGE: self._run_language.value})
            model_info.update({ModelInfoKeys.DRUM_VERSION: drum_version})
            model_info.update({ModelInfoKeys.DRUM_SERVER: "flask"})
            model_info.update(
                {ModelInfoKeys.MODEL_METADATA: read_model_metadata_yaml(self._code_dir)}
            )

            return model_info, HTTP_200_OK

        @model_api.route("/health/", methods=["GET"])
        def health():
            if hasattr(self._predictor, "readiness_probe"):
                return self._predictor.readiness_probe()

            return {"message": "OK"}, HTTP_200_OK

        @model_api.route("/predictions/", methods=["POST"])
        @model_api.route("/predict/", methods=["POST"])
        @model_api.route("/invocations", methods=["POST"])
        def predict():
            logger.debug("Entering predict() endpoint")
            with otel_context(tracer, "drum.invocations", request.headers):
                self._pre_predict_and_transform()
                try:
                    response, response_status = self.do_predict_structured(logger=logger)
                finally:
                    self._post_predict_and_transform()

            return response, response_status

        @model_api.route("/transform/", methods=["POST"])
        def transform():
            logger.debug("Entering transform() endpoint")
            with otel_context(tracer, "drum.transform", request.headers):
                self._pre_predict_and_transform()
                try:
                    response, response_status = self.do_transform(logger=logger)
                finally:
                    self._post_predict_and_transform()

            return response, response_status

        @model_api.route("/predictionsUnstructured/", methods=["POST"])
        @model_api.route("/predictUnstructured/", methods=["POST"])
        def predict_unstructured():
            logger.debug("Entering predict() endpoint")
            with otel_context(tracer, "drum.predictUnstructured", request.headers):
                self._pre_predict_and_transform()
                try:
                    response, response_status = self.do_predict_unstructured(logger=logger)
                finally:
                    self._post_predict_and_transform()
            return (response, response_status)

        # Chat routes are defined without trailing slash because this is required by the OpenAI python client.
        @model_api.route("/chat/completions", methods=["POST"])
        @model_api.route("/v1/chat/completions", methods=["POST"])
        def chat():
            logger.debug("Entering chat endpoint")
            with otel_context(tracer, "drum.chat.completions", request.headers) as span:
                span.set_attributes(extract_chat_request_attributes(request.json))
                self._pre_predict_and_transform()
                try:
                    response, response_status = self.do_chat(logger=logger)
                finally:
                    self._post_predict_and_transform()

                if isinstance(response, dict) and response_status == 200:
                    span.set_attributes(extract_chat_response_attributes(response))

            return response, response_status

        # models routes are defined without trailing slash because this is required by the OpenAI python client.
        @model_api.route("/models", methods=["GET"])
        @model_api.route("/v1/models", methods=["GET"])
        def get_supported_llm_models():
            logger.debug("Entering models endpoint")

            self._pre_predict_and_transform()

            try:
                response, response_status = self.get_supported_llm_models(logger=logger)
            finally:
                self._post_predict_and_transform()

            return response, response_status

        @model_api.route("/directAccess/<path:path>", methods=["GET", "POST", "PUT"])
        @model_api.route("/nim/<path:path>", methods=["GET", "POST", "PUT"])
        def forward_request(path):
            with otel_context(tracer, "drum.directAccess", request.headers) as span:
                if not hasattr(self._predictor, "openai_host") or not hasattr(
                    self._predictor, "openai_port"
                ):
                    msg = "This endpoint is only supported by OpenAI based predictors"
                    span.set_status(StatusCode.ERROR, msg)
                    return {"message": msg}, HTTP_400_BAD_REQUEST

                openai_host = self._predictor.openai_host
                openai_port = self._predictor.openai_port
                resp = requests.request(
                    method=request.method,
                    url=f"http://{openai_host}:{openai_port}/{path.rstrip('/')}",
                    headers=request.headers,
                    params=request.args,
                    timeout=self.get_nim_direct_access_request_timeout(),
                    data=request.get_data(),
                    allow_redirects=False,
                )

            return Response(resp.content, status=resp.status_code, headers=dict(resp.headers))

        @model_api.route("/stats/", methods=["GET"])
        def stats():
            ret_dict = self._resource_monitor.collect_resources_info()

            self._stats_collector.round()
            ret_dict["time_info"] = {}
            for name in self._stats_collector.get_report_names():
                d = self._stats_collector.dict_report(name)
                ret_dict["time_info"][name] = d
            self._stats_collector.stats_reset()
            return ret_dict, HTTP_200_OK

        @model_api.errorhandler(Exception)
        def handle_exception(e):
            logger.exception(e)

            if isinstance(e, HTTPException) and e.code == HTTP_400_BAD_REQUEST:
                return jsonify(error=e.description), e.code

            return {"message": "ERROR: {}".format(e)}, HTTP_500_INTERNAL_SERVER_ERROR

        # Disables warning for development server
        cli = sys.modules["flask.cli"]
        cli.show_server_banner = lambda *x: None

        app = get_flask_app(model_api)
        self.load_flask_extensions(app)
        self._run_flask_app(app, self._terminate)

        if self._stats_collector:
            self._stats_collector.print_reports()

        return []

    def get_gunicorn_config(self):
        config = {}
        if RuntimeParameters.has("DRUM_GUNICORN_WORKER_CLASS"):
            worker_class = str(RuntimeParameters.get("DRUM_GUNICORN_WORKER_CLASS"))
            if worker_class.lower() in {"sync", "gevent"}:
                config["worker_class"] = worker_class

        if RuntimeParameters.has("DRUM_GUNICORN_WORKER_CONNECTIONS"):
            worker_connections = int(RuntimeParameters.get("DRUM_GUNICORN_WORKER_CONNECTIONS"))
            if 1 <= worker_connections <= 10000:
                config["worker_connections"] = worker_connections

        if RuntimeParameters.has("DRUM_GUNICORN_BACKLOG"):
            backlog = int(RuntimeParameters.get("DRUM_GUNICORN_BACKLOG"))
            if 1 <= backlog <= 2048:
                config["backlog"] = backlog

        if RuntimeParameters.has("DRUM_GUNICORN_TIMEOUT"):
            timeout = int(RuntimeParameters.get("DRUM_GUNICORN_TIMEOUT"))
            if 1 <= timeout <= 3600:
                config["timeout"] = timeout

        if RuntimeParameters.has("DRUM_GUNICORN_GRACEFUL_TIMEOUT"):
            graceful_timeout = int(RuntimeParameters.get("DRUM_GUNICORN_GRACEFUL_TIMEOUT"))
            if 1 <= graceful_timeout <= 3600:
                config["graceful_timeout"] = graceful_timeout

        if RuntimeParameters.has("DRUM_GUNICORN_KEEP_ALIVE"):
            keepalive = int(RuntimeParameters.get("DRUM_GUNICORN_KEEP_ALIVE"))
            if 1 <= keepalive <= 3600:
                config["keepalive"] = keepalive

        if RuntimeParameters.has("DRUM_GUNICORN_MAX_REQUESTS"):
            max_requests = int(RuntimeParameters.get("DRUM_GUNICORN_MAX_REQUESTS"))
            if 1 <= max_requests <= 10000:
                config["max_requests"] = max_requests

        if RuntimeParameters.has("DRUM_GUNICORN_MAX_REQUESTS_JITTER"):
            max_requests_jitter = int(RuntimeParameters.get("DRUM_GUNICORN_MAX_REQUESTS_JITTER"))
            if 1 <= max_requests_jitter <= 10000:
                config["max_requests_jitter"] = max_requests_jitter

        if RuntimeParameters.has("DRUM_GUNICORN_LOG_LEVEL"):
            loglevel = str(RuntimeParameters.get("DRUM_GUNICORN_LOG_LEVEL"))
            if loglevel.lower() in {"debug", "info", "warning", "error", "critical"}:
                config["loglevel"] = loglevel

        if RuntimeParameters.has("DRUM_GUNICORN_WORKERS"):
            workers = int(RuntimeParameters.get("DRUM_GUNICORN_WORKERS"))
            if 0 < workers < 200:
                config["workers"] = workers

        return config

    def get_server_type(self):
        server_type = "flask"
        if RuntimeParameters.has("DRUM_SERVER_TYPE"):
            server_type = str(RuntimeParameters.get("DRUM_SERVER_TYPE"))
            if server_type.lower() in {"flask", "gunicorn"}:
                server_type = server_type.lower()
        return server_type

    def _run_flask_app(self, app, termination_hook):
        host = self._params.get("host", None)
        port = self._params.get("port", None)
        server_type = self.get_server_type()
        processes = 1
        if self._params.get("processes"):
            processes = self._params.get("processes")
            logger.info("Number of webserver processes: %s", processes)
        try:
            if RuntimeParameters.has("USE_NIM_WATCHDOG") and str(
                RuntimeParameters.get("USE_NIM_WATCHDOG")
            ).lower() in ["true", "1", "yes"]:
                # Start the watchdog thread before running the app
                self._server_watchdog = Thread(
                    target=self.watchdog,
                    args=(port,),
                    daemon=True,
                    name="NIM Sidecar Watchdog",
                )
                self._server_watchdog.start()

            if server_type == "gunicorn":
                logger.info("Starting gunicorn server")
                try:
                    from gunicorn.app.base import BaseApplication
                except ImportError:
                    BaseApplication = None
                    raise DrumCommonException("gunicorn is not installed. Please install gunicorn.")

                class GunicornApp(BaseApplication):
                    def __init__(self, app, host, port, params, gunicorn_config, termination_hook):
                        self.application = app
                        self.host = host
                        self.port = port
                        self.params = params
                        self.gunicorn_config = gunicorn_config
                        self.termination_hook = termination_hook
                        super().__init__()

                    def load_config(self):
                        self.cfg.set("bind", f"{self.host}:{self.port}")
                        workers = (
                            self.params.get("max_workers")
                            or self.params.get("processes")
                        )
                        if self.gunicorn_config.get("workers"):
                            workers = self.gunicorn_config.get("workers")
                        self.cfg.set("workers", workers)
                        self.cfg.set("reuse_port", True)
                        self.cfg.set("preload_app", True)

                        self.cfg.set(
                            "worker_class", self.gunicorn_config.get("worker_class", "sync")
                        )
                        self.cfg.set("backlog", self.gunicorn_config.get("backlog", 2048))
                        self.cfg.set("timeout", self.gunicorn_config.get("timeout", 120))
                        self.cfg.set(
                            "graceful_timeout", self.gunicorn_config.get("graceful_timeout", 60)
                        )
                        self.cfg.set("keepalive", self.gunicorn_config.get("keepalive", 5))
                        self.cfg.set("max_requests", self.gunicorn_config.get("max_requests", 1000))
                        self.cfg.set(
                            "max_requests_jitter",
                            self.gunicorn_config.get("max_requests_jitter", 500),
                        )

                        if self.gunicorn_config.get("worker_connections"):
                            self.cfg.set(
                                "worker_connections", self.gunicorn_config.get("worker_connections")
                            )
                        self.cfg.set("loglevel", self.gunicorn_config.get("loglevel", "info"))

                        # Properly assign the worker_exit hook
                        if self.termination_hook:
                            self.cfg.set("worker_exit", self._worker_exit_hook)

                        '''self.cfg.set("accesslog", "-")
                        self.cfg.set("errorlog", "-")  # if you want error logs to stdout
                        self.cfg.set(
                            "access_log_format",
                            '%(t)s %(h)s %(l)s %(u)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"',
                        )'''
                        # Remove unsupported config keys: access_logfile, error_logfile, access_logformat
                        # These must be set via CLI, not config API

                    def load(self):
                        return self.application

                    def _worker_exit_hook(self, server, worker):
                        pid = worker.pid
                        server.log.info(f"[HOOK] Worker PID {pid} exiting — running termination hook.")

                        def run_hook():
                            try:
                                self.termination_hook()
                                server.log.info(f"[HOOK] Worker PID {pid} termination logic completed.")
                            except Exception as e:
                                server.log.error(f"[HOOK ERROR] Worker PID {pid}: {e}")

                            try:
                                # 🔍 Найти процесс, который слушает тот же порт
                                port = self.port
                                occupying_proc = None

                                for proc in psutil.process_iter(['pid', 'name']):
                                    try:
                                        for conn in proc.connections(kind='inet'):
                                            if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port:
                                                occupying_proc = proc
                                                break
                                        if occupying_proc:
                                            break
                                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                                        continue

                                if occupying_proc:
                                    server.log.info(
                                        f"[PORT OCCUPIED] Порт {port} занят процессом: PID={occupying_proc.pid}, имя={occupying_proc.name()}"
                                    )
                                else:
                                    server.log.info(f"[PORT FREE] Порт {port} свободен.")

                                server.log.info(f"[HOOK] Worker PID {pid} termination logic completed.")
                            except Exception as e:
                                server.log.error(f"[HOOK ERROR] Worker PID {pid}: {e}")



                        thread = threading.Thread(target=run_hook)
                        thread.start()


                        '''for thread in threads:
                            try:
                                server.log.info(f"Name: {thread.name}, ID: {thread.ident}, Daemon: {thread.daemon}")
                            '''
                        #server.log.info(f"Active thread count:", threading.active_count())
                        thread.join(timeout=20)
                        server.log.info(f"[HOOK] Worker PID {pid} cleanup done or timed out.")

                gunicorn_config = self.get_gunicorn_config()
                GunicornApp(app, host, port, self._params, gunicorn_config, termination_hook).run()
            else:
                # Configure the server with timeout settings
                app.run(
                    host=host,
                    port=port,
                    threaded=False,
                    processes=processes,
                    **(
                        {"request_handler": TimeoutWSGIRequestHandler}
                        if RuntimeParameters.has("DRUM_CLIENT_REQUEST_TIMEOUT")
                        else {}
                    ),
                )
        except OSError as e:
            raise DrumCommonException(f"{e}: host: {host}; port: {port}")

    def _kill_all_processes(self):
        """
        Forcefully terminates all running processes related to the server.
        Attempts a clean termination first, then uses system commands to kill remaining processes.
        Logs errors encountered during termination.
        """

        logger.error("All health check attempts failed. Forcefully killing all processes.")

        # First try clean termination
        try:
            self._terminate()
        except Exception as e:
            logger.error(f"Error during clean termination: {str(e)}")

        # Use more direct system commands to kill processes
        try:
            # Kill packedge jobs first (more aggressive approach)
            logger.info("Killing Python package jobs")
            # Run `busybox ps` and capture output
            result = subprocess.run(["busybox", "ps"], capture_output=True, text=True)
            # Parse lines, skip the header
            lines = result.stdout.strip().split("\n")[1:]
            # Extract the PID (first column)
            pids = [int(line.split()[0]) for line in lines]
            for pid in pids:
                print("Killing pid:", pid)
                os.kill(pid, signal.SIGTERM)
        except Exception as kill_error:
            logger.error(f"Error during process killing: {str(kill_error)}")

    def watchdog(self, port):
        """
        Watchdog thread that periodically checks if the server is alive by making
        GET requests to the /info/ endpoint. Makes 3 attempts with quadratic backoff
        before terminating the Flask app.
        """

        logger.info("Starting watchdog to monitor server health...")

        import os

        url_host = os.environ.get("TEST_URL_HOST", "localhost")
        url_prefix = os.environ.get(URL_PREFIX_ENV_VAR_NAME, "")
        health_url = f"http://{url_host}:{port}{url_prefix}/info/"

        request_timeout = 120
        if RuntimeParameters.has("NIM_WATCHDOG_REQUEST_TIMEOUT"):
            try:
                request_timeout = int(RuntimeParameters.get("NIM_WATCHDOG_REQUEST_TIMEOUT"))
            except ValueError:
                logger.warning(
                    "Invalid value for NIM_WATCHDOG_REQUEST_TIMEOUT, using default of 120 seconds"
                )
        logger.info("Nim watchdog health check request timeout is %s", request_timeout)
        check_interval = 10  # seconds
        max_attempts = 3
        if RuntimeParameters.has("NIM_WATCHDOG_MAX_ATTEMPTS"):
            try:
                max_attempts = int(RuntimeParameters.get("NIM_WATCHDOG_MAX_ATTEMPTS"))
            except ValueError:
                logger.warning("Invalid value for NIM_WATCHDOG_MAX_ATTEMPTS, using default of 3")
        logger.info("Nim watchdog max attempts: %s", max_attempts)
        attempt = 0
        base_sleep_time = 4

        while True:
            try:
                # Check if server is responding to health checks
                logger.debug(f"Server health check")
                response = requests.get(health_url, timeout=request_timeout)
                logger.debug(f"Server health check status: {response.status_code}")
                # Connection succeeded, reset attempts and wait for next check
                attempt = 0
                time.sleep(check_interval)  # Regular check interval
                continue

            except Exception as e:
                attempt += 1
                logger.warning(f"health_url {health_url}")
                logger.warning(
                    f"Server health check failed (attempt {attempt}/{max_attempts}): {str(e)}"
                )

                if attempt >= max_attempts:
                    self._kill_all_processes()

                # Quadratic backoff
                sleep_time = base_sleep_time * (attempt**2)
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)

    def terminate(self):
        terminate_op = getattr(self._predictor, "terminate", None)
        if callable(terminate_op):
            terminate_op()

    def load_flask_extensions(self, app):
        custom_file_paths = list(Path(self._code_dir).rglob("{}.py".format(FLASK_EXT_FILE_NAME)))
        if len(custom_file_paths) > 1:
            raise RuntimeError("Found too many custom hook files: {}".format(custom_file_paths))

        if len(custom_file_paths) == 0:
            logger.info("No %s.py file detected in %s", FLASK_EXT_FILE_NAME, self._code_dir)
            return

        custom_file_path = custom_file_paths[0]
        logger.info("Detected %s .. trying to load Flask extensions", custom_file_path)
        sys.path.insert(0, str(custom_file_path.parent))

        try:
            custom_module = __import__(FLASK_EXT_FILE_NAME)
            custom_module.init_app(app)
        except ImportError as e:
            logger.error("Could not load hooks", exc_info=True)
            raise DrumCommonException(
                "Failed to extend Flask app from [{}] : {}".format(custom_file_path, e)
            )
