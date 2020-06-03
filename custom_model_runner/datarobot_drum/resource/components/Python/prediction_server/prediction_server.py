import logging
import os
import pandas as pd

from flask import Flask, request
from datarobot_drum.resource.components.Python.external_runner.external_runner import ExternalRunner
from datarobot_drum.drum.common import LOGGER_NAME_PREFIX
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.profiler.stats_collector import StatsCollector, StatsOperation
from datarobot_drum.drum.memory_monitor import MemoryMonitor
from mlpiper.common.byte_conv import ByteConv


logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

HTTP_200_OK = 200
HTTP_422_UNPROCESSABLE_ENTITY = 422


class PredictionServer(ExternalRunner):
    def __init__(self, engine):
        super(PredictionServer, self).__init__(engine)
        self._show_perf = False
        self._stats_collector = None
        self._memory_monitor = None

    def configure(self, params):
        super(PredictionServer, self).configure(params)
        self._threaded = self._params.get("threaded", False)
        self._show_perf = self._params.get("show_perf")
        self._stats_collector = StatsCollector(disable_instance=not self._show_perf)

        self._stats_collector.register_report(
            "set_in_df_total", "set_in_df", StatsOperation.SUB, "start"
        )
        self._stats_collector.register_report(
            "run_pipeline_total", "run_pipeline", StatsOperation.SUB, "set_in_df"
        )
        self._stats_collector.register_report(
            "get_out_df_total", "get_out_df", StatsOperation.SUB, "run_pipeline"
        )
        self._memory_monitor = MemoryMonitor()

    def _materialize(self, parent_data_objs, user_data):
        app = Flask(__name__)
        logging.getLogger("werkzeug").setLevel(logger.getEffectiveLevel())
        url_prefix = os.environ.get("URL_PREFIX", "")

        @app.route("{}/predict/".format(url_prefix), methods=["POST"])
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
            self._set_in_df(in_df)
            self._stats_collector.mark("set_in_df")
            self._run_pipeline()
            self._stats_collector.mark("run_pipeline")
            out_df = self._get_out_df()
            self._clean_out_mem()
            self._stats_collector.mark("get_out_df")
            self._stats_collector.disable()

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

            return response_json, response_status

        @app.route("{}/stats/".format(url_prefix))
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

        def _shutdown_server():
            func = request.environ.get("werkzeug.server.shutdown")
            if func is None:
                raise RuntimeError("Not running with the Werkzeug Server")
            func()

        @app.route("{}/".format(url_prefix))
        def ping():
            """This route is used to ensure that server has started"""
            return "Server is up!\n", HTTP_200_OK

        def _shutdown_server():
            func = request.environ.get("werkzeug.server.shutdown")
            if func is None:
                raise RuntimeError("Not running with the Werkzeug Server")
            func()

        @app.route("{}/shutdown/".format(url_prefix), methods=["POST"])
        def shutdown():
            _shutdown_server()
            return "Server shutting down...", HTTP_200_OK

        host = self._params.get("host", None)
        port = self._params.get("port", None)
        try:
            app.run(host, port, threaded=self._threaded)
        except OSError as e:
            raise DrumCommonException("{}: host: {}; port: {}".format(e, host, port))

        self._cleanup_pipeline()
        if self._stats_collector:
            self._stats_collector.print_reports()

        return []
