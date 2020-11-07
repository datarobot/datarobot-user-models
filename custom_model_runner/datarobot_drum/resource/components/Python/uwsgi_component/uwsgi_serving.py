import logging
import os
import sys

from mlpiper.components.restful.flask_route import FlaskRoute
from mlpiper.components.restful_component import RESTfulComponent
from mlpiper.components.restful.metric import Metric, MetricType, MetricRelation

from datarobot_drum.drum.common import (
    RunLanguage,
    URL_PREFIX_ENV_VAR_NAME,
    TARGET_TYPE_ARG_KEYWORD,
    make_predictor_capabilities,
)
from datarobot_drum.profiler.stats_collector import StatsCollector, StatsOperation

from datarobot_drum.drum.server import (
    HTTP_200_OK,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_513_DRUM_PIPELINE_ERROR,
)
from datarobot_drum.drum.memory_monitor import MemoryMonitor
from datarobot_drum.resource.predict_mixin import PredictMixin


class UwsgiServing(RESTfulComponent, PredictMixin):
    def __init__(self, engine):
        super(UwsgiServing, self).__init__(engine)
        self._show_perf = False
        self._stats_collector = None
        self._memory_monitor = None
        self._run_language = None
        self._predictor = None
        self._target_type = None

        self._predict_calls_count = 0

        self._verbose = self._logger.isEnabledFor(logging.DEBUG)

        self._total_predict_requests = Metric(
            "mlpiper.restful.predict_requests",
            title="Total number of stat requests",
            metric_type=MetricType.COUNTER,
            value_type=int,
            metric_relation=MetricRelation.SUM_OF,
        )
        self._error_response = None

    def get_info(self):
        return {
            "python": "{}.{}.{}".format(
                sys.version_info[0], sys.version_info[1], sys.version_info[2]
            ),
            "worker_id": self.get_wid(),
        }

    def configure(self, params):
        """
        @brief      It is called in within the 'deputy' context
        """
        super(UwsgiServing, self).configure(params)
        self._show_perf = self._params.get("show_perf")
        self._run_language = RunLanguage(params.get("run_language"))
        self._target_type = params[TARGET_TYPE_ARG_KEYWORD]

        self._stats_collector = StatsCollector(disable_instance=not self._show_perf)

        self._stats_collector.register_report(
            "run_predictor_total", "finish", StatsOperation.SUB, "start"
        )
        self._memory_monitor = MemoryMonitor()

        self._logger.info(
            "Configure component with input params, name: {}, params: {}".format(
                self.name(), params
            )
        )

    def load_model_callback(self, model_path, stream, version):
        self._logger.info(self.get_info())

        try:
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
                from datarobot_drum.drum.language_predictors.r_predictor.r_predictor import (
                    RPredictor,
                )

                self._predictor = RPredictor()
            self._predictor.configure(self._params)
        except Exception as e:
            self._error_response = {"message": "ERROR: {}".format(e)}

    @FlaskRoute("{}/".format(os.environ.get(URL_PREFIX_ENV_VAR_NAME, "")), methods=["GET"])
    def ping(self, url_params, form_params):
        return HTTP_200_OK, {"message": "OK"}

    @FlaskRoute(
        "{}/capabilities/".format(os.environ.get(URL_PREFIX_ENV_VAR_NAME, "")), methods=["GET"]
    )
    def capabilities(self, url_params, form_params):
        return HTTP_200_OK, make_predictor_capabilities(self._predictor.supported_payload_formats)

    @FlaskRoute("{}/health/".format(os.environ.get(URL_PREFIX_ENV_VAR_NAME, "")), methods=["GET"])
    def health(self, url_params, form_params):
        if self._error_response:
            return HTTP_513_DRUM_PIPELINE_ERROR, self._error_response
        else:
            return HTTP_200_OK, {"message": "OK"}

    @FlaskRoute("{}/stats/".format(os.environ.get(URL_PREFIX_ENV_VAR_NAME, "")), methods=["GET"])
    def prediction_server_stats(self, url_params, form_params):
        mem_info = self._memory_monitor.collect_memory_info()
        ret_dict = {"mem_info": mem_info._asdict()}

        self._stats_collector.round()
        ret_dict["time_info"] = {}
        for name in self._stats_collector.get_report_names():
            d = self._stats_collector.dict_report(name)
            ret_dict["time_info"][name] = d
        self._stats_collector.stats_reset()
        return HTTP_200_OK, ret_dict

    @FlaskRoute("{}/predict/".format(os.environ.get(URL_PREFIX_ENV_VAR_NAME, "")), methods=["POST"])
    def predict(self, url_params, form_params):
        if self._error_response:
            return HTTP_513_DRUM_PIPELINE_ERROR, self._error_response

        self._stats_collector.enable()
        self._stats_collector.mark("start")

        try:
            response, response_status = self.do_predict()

            if response_status == HTTP_200_OK:
                # this counter is managed by uwsgi
                self._total_predict_requests.increase()
                self._predict_calls_count += 1
        except Exception as ex:
            response_status, response = self._handle_exception(ex)
        finally:
            self._stats_collector.mark("finish")
            self._stats_collector.disable()
        return response_status, response

    @FlaskRoute(
        "{}/predictUnstructured/".format(os.environ.get(URL_PREFIX_ENV_VAR_NAME, "")),
        methods=["POST"],
    )
    def predict_unstructured(self, url_params, form_params):
        if self._error_response:
            return HTTP_513_DRUM_PIPELINE_ERROR, self._error_response

        self._stats_collector.enable()
        self._stats_collector.mark("start")

        try:
            response, response_status = self.do_predict_unstructured()

            if response_status == HTTP_200_OK:
                # this counter is managed by uwsgi
                self._total_predict_requests.increase()
                self._predict_calls_count += 1
        except Exception as ex:
            response_status, response = self._handle_exception(ex)
        finally:
            self._stats_collector.mark("finish")
            self._stats_collector.disable()
        return response_status, response

    def _handle_exception(self, ex):
        self._logger.error(ex)
        response_status = HTTP_500_INTERNAL_SERVER_ERROR
        response = {"message": "ERROR: {}".format(ex)}
        return response_status, response

    def _get_stats_dict(self):
        return {
            "predict_calls_per_worker": self._predict_calls_count,
            "predict_calls_total": self._total_predict_requests.get(),
        }
