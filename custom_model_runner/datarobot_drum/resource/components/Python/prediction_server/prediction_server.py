import logging
import pandas as pd

from flask import request

from mlpiper.components.connectable_component import ConnectableComponent

from datarobot_drum.drum.common import LOGGER_NAME_PREFIX
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.profiler.stats_collector import StatsCollector, StatsOperation
from datarobot_drum.drum.memory_monitor import MemoryMonitor
from datarobot_drum.drum.common import RunLanguage

from datarobot_drum.drum.server import (
    HTTP_200_OK,
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_500_INTERNAL_SERVER_ERROR,
    get_flask_app,
    base_api_blueprint,
)

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class PredictionServer(ConnectableComponent):
    def __init__(self, engine):
        super(PredictionServer, self).__init__(engine)
        self._show_perf = False
        self._stats_collector = None
        self._memory_monitor = None
        self._run_language = None
        self._predictor = None

    def configure(self, params):
        super(PredictionServer, self).configure(params)
        self._threaded = self._params.get("threaded", False)
        self._show_perf = self._params.get("show_perf")
        self._stats_collector = StatsCollector(disable_instance=not self._show_perf)

        self._stats_collector.register_report(
            "run_predictor_total", "finish", StatsOperation.SUB, "start"
        )
        self._memory_monitor = MemoryMonitor()
        self._run_language = RunLanguage(params.get("run_language"))
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

        @model_api.route("/health/", methods=["GET"])
        def health():
            return {"message": "OK"}, HTTP_200_OK

        @model_api.route("/predict/", methods=["POST"])
        def predict():
            response_status = HTTP_200_OK
            file_key = "X"
            logger.debug("Entering predict() endpoint")
            REGRESSION_PRED_COLUMN = "Predictions"
            filename = request.files[file_key] if file_key in request.files else None
            logger.debug("Filename provided under X key: {}".format(filename))

            if not filename:
                wrong_key_error_message = "Samples should be provided as a csv file under `{}` key.".format(
                    file_key
                )
                logger.error(wrong_key_error_message)
                response_status = HTTP_422_UNPROCESSABLE_ENTITY
                return {"message": "ERROR: " + wrong_key_error_message}, response_status

            in_df = pd.read_csv(filename)

            # TODO labels have to be provided as command line arguments or within configure endpoint
            self._stats_collector.enable()
            self._stats_collector.mark("start")
            out_df = self._predictor.predict(in_df)

            num_columns = len(out_df.columns)
            # float32 is not JSON serializable, so cast to float, which is float64
            out_df = out_df.astype("float")
            if num_columns == 1:
                # df.to_json() is much faster.
                # But as it returns string, we have to assemble final json using strings.
                df_json = out_df[REGRESSION_PRED_COLUMN].to_json(orient="records")
                response_json = '{{"predictions":{df_json}}}'.format(df_json=df_json)
            elif num_columns == 2:
                # df.to_json() is much faster.
                # But as it returns string, we have to assemble final json using strings.
                df_json_str = out_df.to_json(orient="records")
                response_json = '{{"predictions":{df_json}}}'.format(df_json=df_json_str)
            else:
                ret_str = (
                    "Predictions dataframe has {} columns; "
                    "Expected: 1 - for regression, 2 - for binary classification.".format(
                        num_columns
                    )
                )
                response_json = {"message": "ERROR: " + ret_str}
                response_status = HTTP_422_UNPROCESSABLE_ENTITY

            self._stats_collector.mark("finish")
            self._stats_collector.disable()
            return response_json, response_status

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
        logging.getLogger("werkzeug").setLevel(logger.getEffectiveLevel())

        host = self._params.get("host", None)
        port = self._params.get("port", None)
        try:
            app.run(host, port, threaded=self._threaded)
        except OSError as e:
            raise DrumCommonException("{}: host: {}; port: {}".format(e, host, port))

        if self._stats_collector:
            self._stats_collector.print_reports()

        return []
