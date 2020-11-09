import logging
from mlpiper.components.connectable_component import ConnectableComponent

from datarobot_drum.drum.common import LOGGER_NAME_PREFIX, make_predictor_capabilities
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.profiler.stats_collector import StatsCollector, StatsOperation
from datarobot_drum.drum.memory_monitor import MemoryMonitor
from datarobot_drum.drum.common import RunLanguage, TARGET_TYPE_ARG_KEYWORD
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
        self._memory_monitor = None
        self._run_language = None
        self._predictor = None
        self._target_type = None

    def configure(self, params):
        super(PredictionServer, self).configure(params)
        self._show_perf = self._params.get("show_perf")
        self._run_language = RunLanguage(params.get("run_language"))
        self._target_type = params[TARGET_TYPE_ARG_KEYWORD]

        self._stats_collector = StatsCollector(disable_instance=not self._show_perf)

        self._stats_collector.register_report(
            "run_predictor_total", "finish", StatsOperation.SUB, "start"
        )
        self._memory_monitor = MemoryMonitor(monitor_current_process=True)

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
        elif self._run_language == RunLanguage.R:
            # this import is here, because RPredictor imports rpy library,
            # which is not installed for Java and Python cases.
            from datarobot_drum.drum.language_predictors.r_predictor.r_predictor import RPredictor

            self._predictor = RPredictor()
        else:
            raise DrumCommonException(
                "Prediction server doesn't support language: {} ".format(self._run_language)
            )

        self._predictor.configure(params)

    def _materialize(self, parent_data_objs, user_data):
        model_api = base_api_blueprint()

        @model_api.route("/capabilities/", methods=["GET"])
        def capabilities():
            return make_predictor_capabilities(self._predictor.supported_payload_formats)

        @model_api.route("/health/", methods=["GET"])
        def health():
            return {"message": "OK"}, HTTP_200_OK

        @model_api.route("/predict/", methods=["POST"])
        def predict():
            logger.debug("Entering predict() endpoint")

            self._stats_collector.enable()
            self._stats_collector.mark("start")

            try:
                response, response_status = self.do_predict(logger=logger)
            finally:
                self._stats_collector.mark("finish")
                self._stats_collector.disable()
            return response, response_status

        @model_api.route("/predictUnstructured/", methods=["POST"])
        def predict_unstructured():
            logger.debug("Entering predict() endpoint")

            self._stats_collector.enable()
            self._stats_collector.mark("start")

            try:
                response, response_status = self.do_predict_unstructured(logger=logger)
            finally:
                self._stats_collector.mark("finish")
                self._stats_collector.disable()
            return response, response_status

        @model_api.route("/stats/", methods=["GET"])
        def stats():
            mem_info = self._memory_monitor.collect_memory_info()
            ret_dict = {"mem_info": mem_info._asdict()}

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

        app = get_flask_app(model_api)

        host = self._params.get("host", None)
        port = self._params.get("port", None)
        try:
            app.run(host, port, threaded=False)
        except OSError as e:
            raise DrumCommonException("{}: host: {}; port: {}".format(e, host, port))

        if self._stats_collector:
            self._stats_collector.print_reports()

        return []
