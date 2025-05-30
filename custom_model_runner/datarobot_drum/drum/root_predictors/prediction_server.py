"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import sys
from pathlib import Path

import requests
from flask import Response, jsonify, request
from werkzeug.exceptions import HTTPException

from opentelemetry import trace
from datarobot_drum.drum.description import version as drum_version
from datarobot_drum.drum.enum import (
    FLASK_EXT_FILE_NAME,
    GPU_PREDICTORS,
    LOGGER_NAME_PREFIX,
    TARGET_TYPE_ARG_KEYWORD,
    ModelInfoKeys,
    RunLanguage,
    TargetType,
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
        self._run_flask_app(app)

        if self._stats_collector:
            self._stats_collector.print_reports()

        return []

    def _run_flask_app(self, app):
        host = self._params.get("host", None)
        port = self._params.get("port", None)

        processes = 1
        if self._params.get("processes"):
            processes = self._params.get("processes")
            logger.info("Number of webserver processes: %s", processes)
        try:
            app.run(host, port, threaded=False, processes=processes)
        except OSError as e:
            raise DrumCommonException("{}: host: {}; port: {}".format(e, host, port))

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
