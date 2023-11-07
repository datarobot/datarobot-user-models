"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import sys
from pathlib import Path
from mlpiper.components.connectable_component import ConnectableComponent

from datarobot_drum.drum.common import (
    make_predictor_capabilities,
    read_model_metadata_yaml,
)
from datarobot_drum.drum.enum import (
    LOGGER_NAME_PREFIX,
    TARGET_TYPE_ARG_KEYWORD,
    ModelInfoKeys,
    RunLanguage,
    TargetType,
    FLASK_EXT_FILE_NAME,
)
from datarobot_drum.drum.description import version as drum_version
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.profiler.stats_collector import StatsCollector, StatsOperation
from datarobot_drum.drum.resource_monitor import ResourceMonitor

from datarobot_drum.resource.components.Python.prediction_server.stdout_flusher import StdoutFlusher
from datarobot_drum.resource.deployment_config_helpers import parse_validate_deployment_config_file
from datarobot_drum.resource.predict_mixin import PredictMixin


from datarobot_drum.drum.server import (
    HTTP_200_OK,
    HTTP_500_INTERNAL_SERVER_ERROR,
    get_flask_app,
    base_api_blueprint,
)

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class PredictionServer(ConnectableComponent, PredictMixin):
    def __init__(self, engine):
        super(PredictionServer, self).__init__(engine)
        self._show_perf = False
        self._stats_collector = None
        self._resource_monitor = None
        self._run_language = None
        self._predictor = None
        self._target_type = None
        self._code_dir = None
        self._deployment_config = None
        self._stdout_flusher = StdoutFlusher()

    def configure(self, params):
        super(PredictionServer, self).configure(params)
        self._code_dir = self._params.get("__custom_model_path__")
        self._show_perf = self._params.get("show_perf")
        self._run_language = RunLanguage(params.get("run_language"))
        self._target_type = TargetType(params[TARGET_TYPE_ARG_KEYWORD])

        self._stats_collector = StatsCollector(disable_instance=not self._show_perf)

        self._stats_collector.register_report(
            "run_predictor_total", "finish", StatsOperation.SUB, "start"
        )
        self._resource_monitor = ResourceMonitor(monitor_current_process=True)
        self._deployment_config = parse_validate_deployment_config_file(
            self._params["deployment_config"]
        )

        if self._run_language == RunLanguage.PYTHON:
            from datarobot_drum.drum.language_predictors.python_predictor.python_predictor import (
                PythonPredictor,
            )

            self._predictor = PythonPredictor()
        elif self._run_language == RunLanguage.JAVA:
            from datarobot_drum.drum.language_predictors.java_predictor.java_predictor import (
                JavaPredictor,
            )

            self._predictor = JavaPredictor()
        elif self._run_language == RunLanguage.JULIA:
            from datarobot_drum.drum.language_predictors.julia_predictor.julia_predictor import (
                JlPredictor,
            )

            self._predictor = JlPredictor()
        elif self._run_language == RunLanguage.R:
            # this import is here, because RPredictor imports rpy library,
            # which is not installed for Java and Python cases.
            from datarobot_drum.drum.language_predictors.r_predictor.r_predictor import RPredictor

            self._predictor = RPredictor()
        else:
            raise DrumCommonException(
                "Prediction server doesn't support language: {} ".format(self._run_language)
            )

        self._stdout_flusher.start()
        self._predictor.mlpiper_configure(params)

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

    def _materialize(self, parent_data_objs, user_data):
        model_api = base_api_blueprint(self._terminate)

        @model_api.route("/capabilities/", methods=["GET"])
        def capabilities():
            return make_predictor_capabilities(self._predictor.supported_payload_formats)

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
            return {"message": "OK"}, HTTP_200_OK

        @model_api.route("/predictions/", methods=["POST"])
        @model_api.route("/predict/", methods=["POST"])
        def predict():
            logger.debug("Entering predict() endpoint")

            self._pre_predict_and_transform()

            try:
                response, response_status = self.do_predict_structured(logger=logger)
            finally:
                self._post_predict_and_transform()

            return response, response_status

        @model_api.route("/transform/", methods=["POST"])
        def transform():
            logger.debug("Entering transform() endpoint")

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

            self._pre_predict_and_transform()

            try:
                response, response_status = self.do_predict_unstructured(logger=logger)
            finally:
                self._post_predict_and_transform()

            return response, response_status

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
            return {"message": "ERROR: {}".format(e)}, HTTP_500_INTERNAL_SERVER_ERROR

        # Disables warning for development server
        cli = sys.modules["flask.cli"]
        cli.show_server_banner = lambda *x: None

        app = get_flask_app(model_api)
        self.load_flask_extensions(app)

        host = self._params.get("host", None)
        port = self._params.get("port", None)
        try:
            app.run(host, port, threaded=False)
        except OSError as e:
            raise DrumCommonException("{}: host: {}; port: {}".format(e, host, port))

        if self._stats_collector:
            self._stats_collector.print_reports()

        return []

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
