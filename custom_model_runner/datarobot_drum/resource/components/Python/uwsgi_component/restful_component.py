"""
The RESTfulComponent class is used to create RESTful model serving components.
It is an abstract class that is supposed to be extended and the sub-class is
supposed to implement its abstract methods.

The class provide the core functionality of the model restful serving, which
includes setup and preparation of the environment, configurations as well as
executing of required services (.e.g uwsig, nginx)
"""

import abc
import os
import socket
import tempfile
import time
import logging
import subprocess
import sys

from mlpiper.pipeline import java_mapping
from mlpiper.model.model_env import ModelEnv
from mlpiper.ml_engine.rest_model_serving_engine import RestModelServingEngine
from mlpiper.common import constants
from mlpiper.components import parameter
from mlpiper.common.mlpiper_exception import MLPiperException
from mlpiper.components.connectable_component import ConnectableComponent
from mlpiper.components.restful.constants import (
    SharedConstants,
    ComponentConstants,
    UwsgiConstants,
    NginxConstants,
)
from mlpiper.components.restful.constants import RestfulConstants
from mlpiper.components.restful.nginx_broker import NginxBroker
from mlpiper.components.restful.flask_route import FlaskRoute
from mlpiper.components.restful.metric import Metric, MetricType, MetricRelation

from datarobot_drum.resource.components.Python.uwsgi_component.uwsgi_broker import UwsgiBroker


class RESTfulComponent(ConnectableComponent):
    _uuid_engine = None
    _stats_path_filename = None
    _stats_count = 0

    def __init__(self, engine):
        # When this component is created from uwsgi entry point, engine param is None.
        # So check for uwsgi installation only once from pipeline/component creation,
        # when engine is not None.
        if engine is not None:
            if subprocess.run(
                [sys.executable, "-m", "pip", "show", "uwsgi"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            ).returncode != 0:
                err_msg = "'uwsgi' package is required. " \
                          "If you are running on a Windows system, " \
                          "you may not be able to use RESTful component. " \
                          "Check how to install uwsgi on Windows. " \
                          "To install uwsgi on a Mac/Linux system, call " \
                          "`pip install uwsgi` or `pip install mlpiper[uwsgi]`"
                raise MLPiperException(err_msg)

        super(RESTfulComponent, self).__init__(
            engine if engine else RestModelServingEngine("uwsgi-context")
        )
        self._dry_run = False
        self._wsgi_broker = None
        self._nginx_broker = None
        self._wid = None
        self._wuuid = None

        self._total_stat_requests = Metric(
            "mlpiper.restful.stat_requests",
            title="Total number of stat requests",
            metric_type=MetricType.COUNTER,
            value_type=int,
            metric_relation=MetricRelation.SUM_OF,
        )

    def set_wid(self, wid):
        self._wid = wid

    def get_wid(self):
        return self._wid

    def set_wuuid(self, wuuid):
        self._wuuid = wuuid

    def _validate_output(self, objs):
        pass

    def _post_validation(self, objs):
        pass

    def configure(self, params):
        super(RESTfulComponent, self).configure(params)

    def _materialize(self, parent_data_objs, user_data):
        monitor_info = {
            UwsgiConstants.MONITOR_ERROR_KEY: None,
            UwsgiConstants.MONITOR_THREAD_KEY: None,
        }
        self._setup(self._ml_engine.pipeline_name, monitor_info)
        self._wait_and_monitor_errors(monitor_info)

    def _setup(self, pipeline_name, monitor_info):
        target_path = tempfile.mkdtemp(
            dir=ComponentConstants.TMP_RESTFUL_ROOT,
            prefix=ComponentConstants.TMP_RESTFUL_DIR_PREFIX,
        )
        os.chmod(target_path, 0o777)

        fd, stats_path_filename = tempfile.mkstemp(
            dir=ComponentConstants.TMP_RESTFUL_ROOT,
            prefix=ComponentConstants.TMP_RESTFUL_DIR_PREFIX,
        )
        os.chmod(stats_path_filename, 0o777)

        self._logger.debug("Path for stats {}".format(stats_path_filename))

        shared_conf = {
            SharedConstants.TARGET_PATH_KEY: target_path,
            SharedConstants.SOCK_FILENAME_KEY: UwsgiConstants.SOCK_FILENAME,
            SharedConstants.STATS_SOCK_FILENAME_KEY: UwsgiConstants.STATS_SOCK_FILENAME,
            SharedConstants.STANDALONE: self._ml_engine.standalone,
            SharedConstants.STATS_PATH_FILENAME_KEY: stats_path_filename,
        }

        log_format = self._params.get(
            ComponentConstants.LOG_FORMAT_KEY, ComponentConstants.DEFAULT_LOG_FORMAT
        )

        log_level_param = self._params.get(
            ComponentConstants.LOG_LEVEL_KEY, ComponentConstants.DEFAULT_LOG_LEVEL
        ).lower()
        log_level = constants.LOG_LEVELS.get(log_level_param, logging.INFO)
        self._logger.debug(
            "log_level_param: {}, log_level: {}, level_constants: {}".format(
                log_level_param, log_level, constants.LOG_LEVELS
            )
        )

        stats_reporting_interval_sec = self._params.get(
            ComponentConstants.STATS_REPORTING_INTERVAL_SEC,
            ComponentConstants.DEFAULT_STATS_REPORTING_INTERVAL_SEC,
        )

        model_filepath_key = java_mapping.RESERVED_KEYS[
            ComponentConstants.INPUT_MODEL_TAG_NAME
        ]
        self._params[model_filepath_key] = ModelEnv(
            self._params[model_filepath_key], self._ml_engine.standalone
        ).model_filepath

        single_uwsgi_worker = parameter.str2bool(
            self._params.get(ComponentConstants.SINGLE_UWSGI_WORKER_KEY, False)
        )

        # Used for uwsgi UDP logging
        logging_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logging_socket.bind(("", 0))
        logging_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        uwsgi_entry_point_conf = {
            UwsgiConstants.RESTFUL_COMP_MODULE_KEY: self.__module__,
            UwsgiConstants.RESTFUL_COMP_CLS_KEY: self.__class__.__name__,
            UwsgiConstants.LOGGING_UDP_SOCKET: logging_socket,
            ComponentConstants.LOG_FORMAT_KEY: log_format,
            ComponentConstants.LOG_LEVEL_KEY: log_level,
            ComponentConstants.STATS_REPORTING_INTERVAL_SEC: stats_reporting_interval_sec,
            UwsgiConstants.PARAMS_KEY: self._params,
            UwsgiConstants.PIPELINE_NAME_KEY: pipeline_name,
            UwsgiConstants.MODEL_PATH_KEY: self._params[model_filepath_key],
            UwsgiConstants.DEPUTY_ID_KEY: self._ml_engine.get_uuid(),
            ComponentConstants.UWSGI_DISABLE_LOGGING_KEY: parameter.str2bool(
                self._params.get(
                    ComponentConstants.UWSGI_DISABLE_LOGGING_KEY,
                    ComponentConstants.DEFAULT_UWSGI_DISABLE_LOGGING,
                )
            ),
            ComponentConstants.METRICS_KEY: Metric.metrics(),
            ComponentConstants.SINGLE_UWSGI_WORKER_KEY: single_uwsgi_worker,
        }
        self._logger.debug("uwsgi_entry_point_conf: {}".format(uwsgi_entry_point_conf))

        nginx_conf = {
            ComponentConstants.HOST_KEY: ComponentConstants.DEFAULT_HOST,
            ComponentConstants.PORT_KEY: self._params[ComponentConstants.PORT_KEY],
            NginxConstants.DISABLE_ACCESS_LOG_KEY: log_level != logging.DEBUG,
        }
        self._logger.debug("nginx_conf: {}".format(nginx_conf))

        self._dry_run = parameter.str2bool(
            self._params.get(
                ComponentConstants.DRY_RUN_KEY, ComponentConstants.DEFAULT_DRY_RUN
            )
        )
        if self._dry_run:
            self._logger.warning(
                "\n\n"
                + 80 * "#"
                + "\n"
                + 25 * " "
                + "Running in DRY RUN mode\n"
                + 80 * "#"
            )

        self._dry_run = parameter.str2bool(
            self._params.get(
                ComponentConstants.DRY_RUN_KEY, ComponentConstants.DEFAULT_DRY_RUN
            )
        )

        self._wsgi_broker = UwsgiBroker(self._ml_engine, self._dry_run).setup_and_run(
            shared_conf, uwsgi_entry_point_conf, monitor_info
        )

        self._nginx_broker = NginxBroker(self._ml_engine, self._dry_run).setup_and_run(
            shared_conf, nginx_conf
        )

    def _wait_and_monitor_errors(self, monitor_info):
        self._logger.info(
            "Going to read model / stop events ... (kidding, going to sleep forever ...)"
        )

        if not self._dry_run and monitor_info[UwsgiConstants.MONITOR_THREAD_KEY]:
            try:
                monitor_info[UwsgiConstants.MONITOR_THREAD_KEY].join()

                if monitor_info[UwsgiConstants.MONITOR_ERROR_KEY]:
                    self._logger.error(monitor_info[UwsgiConstants.MONITOR_ERROR_KEY])
                    raise MLPiperException(
                        monitor_info[UwsgiConstants.MONITOR_ERROR_KEY]
                    )
            except KeyboardInterrupt:
                # When running from mlpiper tool (standalone)
                pass
            finally:
                self._nginx_broker.quit()
                self._wsgi_broker.quit()
        else:
            while True:
                time.sleep(3600 * 24 * 365)

    @abc.abstractmethod
    def load_model_callback(self, model_path, stream, version):
        """
        This abstract method is called whenever a new model is supposed to be loaded. The user
        is responsible to reload the model and start using it in any consequent predictions

        :param model_path: an absolute file path to the model
        """
        pass

    def _on_exit(self):
        cleanup_op = getattr(self, ComponentConstants.CLEANUP_CALLBACK_FUNC_NAME, None)
        if callable(cleanup_op):
            cleanup_op()
        else:
            self._logger.info(
                "'{}' function is not defined by the restful child component!".format(
                    ComponentConstants.CLEANUP_CALLBACK_FUNC_NAME
                )
            )

    @classmethod
    def run(cls, port, model_path):
        raise MLPiperException(
            "Running restful components from CLI is not allowed without mlpiper"
        )

    def _get_stats_dict(self):
        return None

    # NOTE: do not rename this route or over-ride it
    @FlaskRoute("/{}/".format(RestfulConstants.STATS_ROUTE))
    def stats(self, url_params, form_params):
        status_code = 200

        import os
        import json

        stats_dict = {}
        self._stats_count += 1
        self._total_stat_requests.increase()

        if (
            self._stats_path_filename is None
            or os.stat(self._stats_path_filename).st_size == 0
        ):
            pass
        else:
            with open(self._stats_path_filename, "r") as input:
                dict_json = ""
                for line in input:
                    dict_json += line
                try:
                    stats_dict = json.loads(dict_json)
                except Exception as e:
                    stats_dict[RestfulConstants.STATS_SYSTEM_ERROR] = str(e)
                    # stats_dict['system_error'] = str(e)
                    print("Unexpected error: {}", str(e))

        stats_dict[RestfulConstants.STATS_SYSTEM_INFO] = {}
        stats_dict[RestfulConstants.STATS_SYSTEM_INFO][
            RestfulConstants.STATS_WID
        ] = self._wid
        stats_dict[RestfulConstants.STATS_SYSTEM_INFO][
            RestfulConstants.STATS_UUID
        ] = self._uuid_engine
        stats_dict[RestfulConstants.STATS_SYSTEM_INFO][
            RestfulConstants.STATS_WUUID
        ] = self._wuuid

        custom_stats = self._get_stats_dict()
        if custom_stats:
            stats_dict.update(custom_stats)

        return status_code, stats_dict


if __name__ == "__main__":

    class MyResfulComp(RESTfulComponent):
        def __init__(self, engine):
            super(RESTfulComponent, self).__init__(engine)

        def load_model(self, model_path):
            print("Model is reloading, path: {}".format(model_path))

        @FlaskRoute("/v1/predict")
        def predict_v1(self, url_params, form_params):
            print("url_params: {}".format(url_params))
            print("form_params: {}".format(form_params))

            status_code = 200
            dict_response = {"user": "jon", "age": 100}
            return (status_code, dict_response)

        @FlaskRoute("/v2/predict")
        def predict_v2(self, url_params, form_params):
            print("url_params: {}".format(url_params))
            print("form_params: {}".format(form_params))

            status_code = 200
            dict_response = {"user": "jon2", "age": 200}
            return (status_code, dict_response)

    MyResfulComp.run(9999, "/tmp/model_path/xxx.txt")
