"""
For internal use only. The uwsgi borker is designed to handle any related 'uWSGI' actions, such as
setup, configuration and execution
"""
import logging
import os
import re
import subprocess
import sys

from mlpiper.common import constants
from mlpiper.common.base import Base
from mlpiper.common.mlpiper_exception import MLPiperException
from mlpiper.components.restful import util
from mlpiper.components.restful.flask_app_wrapper import FlaskAppWrapper
from mlpiper.components.restful.flask_route import FlaskRoute
from mlpiper.components.restful.flask_custom_json_encoder import FlaskCustomJsonEncode
from mlpiper.components.restful.constants import (
    SharedConstants,
    ComponentConstants,
    UwsgiConstants,
)
from mlpiper.components.restful.uwsgi_monitor import WsgiMonitor
from mlpiper.components.restful.uwsgi_ini_template import WSGI_INI_CONTENT
from mlpiper.components.restful.uwsgi_entry_point_script_template import (
    WSGI_ENTRY_SCRIPT,
)
from mlpiper.components.restful.uwsgi_cheaper_subsystem import UwsgiCheaperSubSystem
from mlpiper.model.model_selector import ModelSelector

try:
    from mlpiper.components.restful.uwsgi_post_fork import UwsgiPostFork
    import uwsgi
except ImportError:
    # You're actually not running under uWSGI
    pass


class UwsgiBroker(Base):
    _restful_comp = None
    _model_selector = None
    _application = None
    _wid = None
    w_logger = None

    #######################################################################
    # Methods that are accessed from the Pipeline RESTful component

    def __init__(self, ml_engine, dry_run=False):
        super(UwsgiBroker, self).__init__()
        self.set_logger(ml_engine.get_engine_logger(self.logger_name()))
        self._ml_engine = ml_engine
        self._dry_run = dry_run
        self._monitor_info = None
        self._target_path = None
        self._monitor = None

    def setup_and_run(self, shared_conf, entry_point_conf, monitor_info):
        self._target_path = shared_conf[SharedConstants.TARGET_PATH_KEY]
        self._logger.info(
            "Setup 'uwsgi' server, target path: {}".format(self._target_path)
        )
        self._monitor_info = monitor_info
        self._verify_dependencies()
        self._generate_entry_point_script(shared_conf, entry_point_conf)
        ini_filepath = self._generate_ini_file(shared_conf, entry_point_conf)
        self._run(shared_conf, entry_point_conf, ini_filepath)
        return self

    def quit(self):
        if not self._dry_run and self._target_path:
            pid_filepath = os.path.join(self._target_path, UwsgiConstants.PID_FILENAME)
            if os.path.isfile(pid_filepath):
                self._logger.info("Stopping uwsgi process ...")
                try:
                    cmd = UwsgiConstants.STOP_CMD.format(pid_filepath=pid_filepath)
                    subprocess.call(cmd, shell=True)
                except subprocess.CalledProcessError as ex:
                    self._logger.info(
                        "'uwsgi' process stopping issue! output: {}, return-code: {}".format(
                            ex.output, ex.returncode
                        )
                    )

    def _verify_dependencies(self):
        util.verify_tool_installation(
            ver_cmd=UwsgiConstants.VER_CMD,
            dev_against_ver=UwsgiConstants.DEV_AGAINST_VERSION,
            logger=self._logger,
        )

    def _generate_entry_point_script(self, shared_conf, entry_point_conf):
        self._logger.debug("shared conf {}".format(shared_conf))
        self._logger.debug("entry conf {}".format(entry_point_conf))

        wsgi_entry_script_code = WSGI_ENTRY_SCRIPT.format(
            module=self.__class__.__module__,
            cls=self.__class__.__name__,
            restful_comp_module=entry_point_conf[
                UwsgiConstants.RESTFUL_COMP_MODULE_KEY
            ],
            restful_comp_cls=entry_point_conf[UwsgiConstants.RESTFUL_COMP_CLS_KEY],
            root_logger_name=constants.LOGGER_NAME_PREFIX,
            log_format=entry_point_conf[ComponentConstants.LOG_FORMAT_KEY],
            log_level=entry_point_conf[ComponentConstants.LOG_LEVEL_KEY],
            params=entry_point_conf[UwsgiConstants.PARAMS_KEY],
            pipeline_name=entry_point_conf[UwsgiConstants.PIPELINE_NAME_KEY],
            model_path=entry_point_conf[UwsgiConstants.MODEL_PATH_KEY],
            standalone=shared_conf[SharedConstants.STANDALONE],
            deputy_id=entry_point_conf[UwsgiConstants.DEPUTY_ID_KEY],
            stats_path_filename=shared_conf[SharedConstants.STATS_PATH_FILENAME_KEY],
            within_uwsgi_context=(not entry_point_conf[ComponentConstants.SINGLE_UWSGI_WORKER_KEY]),
        )

        uwsgi_script_filepath = os.path.join(
            self._target_path, UwsgiConstants.ENTRY_POINT_SCRIPT_NAME
        )
        self._logger.info(
            "Writing uWSGI entry point to: {}".format(uwsgi_script_filepath)
        )

        with open(uwsgi_script_filepath, "w") as f:
            f.write(wsgi_entry_script_code)

    def _generate_ini_file(self, shared_conf, entry_point_conf):
        python_paths = [p for p in sys.path if UwsgiBroker._include_py_path(p)]
        python_paths = ":".join(python_paths)

        cheaper_conf = UwsgiCheaperSubSystem.get_config()
        self._logger.info("Cheaper subsystem: {}".format(cheaper_conf))

        metrics = entry_point_conf[ComponentConstants.METRICS_KEY]

        single_uwsgi_worker = entry_point_conf[ComponentConstants.SINGLE_UWSGI_WORKER_KEY]
        comment_out_master = "# " if single_uwsgi_worker else ""
        comment_out_non_master = "" if single_uwsgi_worker else "# "
        comment_out_cheaper = "# " if single_uwsgi_worker else ""

        # define cheaper system memory limits
        cheaper_memory_limits = ""
        cheaper_rss_limit_soft = entry_point_conf[UwsgiConstants.PARAMS_KEY] \
            .get(ComponentConstants.UWSGI_CHEAPER_RSS_LIMIT_SOFT_KEY, None)
        cheaper_rss_limit_hard = entry_point_conf[UwsgiConstants.PARAMS_KEY] \
            .get(ComponentConstants.UWSGI_CHEAPER_RSS_LIMIT_HARD_KEY, None)
        memory_report = entry_point_conf[UwsgiConstants.PARAMS_KEY] \
            .get(ComponentConstants.UWSGI_MEMORY_REPORT_KEY, False)

        if cheaper_rss_limit_soft:
            cheaper_memory_limits += "cheaper-rss-limit-soft = {}\n".format(cheaper_rss_limit_soft)
        if cheaper_rss_limit_hard:
            cheaper_memory_limits += "cheaper-rss-limit-hard = {}\n".format(cheaper_rss_limit_hard)
        if memory_report or len(cheaper_memory_limits):
            cheaper_memory_limits += "memory-report = true\n" \

        max_workers_from_config = entry_point_conf[UwsgiConstants.PARAMS_KEY] \
            .get(ComponentConstants.UWSGI_MAX_WORKERS_KEY, None)

        workers = 1 if single_uwsgi_worker else cheaper_conf[UwsgiCheaperSubSystem.WORKERS]
        cheaper = cheaper_conf[UwsgiCheaperSubSystem.CHEAPER]
        cheaper_initial = cheaper_conf[UwsgiCheaperSubSystem.CHEAPER_INITIAL]
        cheaper_step = cheaper_conf[UwsgiCheaperSubSystem.CHEAPER_STEP]

        if max_workers_from_config:
            workers = min(max_workers_from_config, workers)
            # cheaper param must be less than workers
            cheaper = min(cheaper, workers - 1)
            cheaper_initial = min(max_workers_from_config, cheaper_initial)
            cheaper_step = min(max_workers_from_config, cheaper_step)

        logging_udp_port = entry_point_conf[UwsgiConstants.LOGGING_UDP_SOCKET].getsockname()[1]
        logto = "localhost:{}".format(logging_udp_port)
        log_socket = "socket:" + logto

        ini_content = WSGI_INI_CONTENT.format(
            restful_app_folder=self._target_path,
            pid_filename=UwsgiConstants.PID_FILENAME,
            sock_filename=shared_conf[SharedConstants.SOCK_FILENAME_KEY],
            stats_sock_filename=shared_conf[SharedConstants.STATS_SOCK_FILENAME_KEY],
            restful_app_file=UwsgiConstants.ENTRY_POINT_SCRIPT_NAME,
            callable_app="application",
            python_paths=python_paths,
            disable_logging=entry_point_conf[
                ComponentConstants.UWSGI_DISABLE_LOGGING_KEY
            ],
            cheaper_memory_limits=cheaper_memory_limits,
            workers=workers,
            cheaper=cheaper,
            cheaper_initial=cheaper_initial,
            cheaper_step=cheaper_step,
            enable_metrics="true" if metrics else "false",
            metrics=self._get_metrics_configuration(metrics),
            log_socket=log_socket,
            logto=logto,
            master="false" if single_uwsgi_worker else "true",
            log_master="false" if single_uwsgi_worker else "true",
            comment_out_master=comment_out_master,
            comment_out_non_master=comment_out_non_master,
            comment_out_cheaper=comment_out_cheaper,
        )

        ini_filepath = os.path.join(self._target_path, UwsgiConstants.INI_FILENAME)
        self._logger.info("Writing uWSGI ini file to: {}".format(ini_filepath))

        with open(ini_filepath, "w") as f:
            f.write(ini_content)

        return ini_filepath

    @staticmethod
    def _include_py_path(filename):
        if filename.endswith(".egg"):
            py_ver = "py{}".format(sys.version_info[0])
            if py_ver in filename:
                return True
            elif not re.search("py[2-3]", filename):
                # 'py2' or 'py3' are not in the egg filename, so include it
                return True
            else:
                return False
        return True

    def _get_metrics_configuration(self, metrics):
        metrics_conf = ""
        if metrics:
            for index, metric_name in enumerate(metrics):
                metrics_conf += (
                    ComponentConstants.METRIC_TEMPLATE.format(metric_name, index + 1)
                    + "\n"
                )
        return metrics_conf

    def _run(self, shared_conf, entry_point_conf, ini_filepath):
        uwsgi_start_cmd = UwsgiConstants.START_CMD.format(filepath=ini_filepath)
        self._logger.info("Running 'uwsgi' server, cmd: '{}'".format(uwsgi_start_cmd))

        if self._dry_run:
            return

        self._monitor = WsgiMonitor(
            self._ml_engine, self._monitor_info, shared_conf, entry_point_conf
        )
        self._logger.debug("Broker uuid: {}".format(self._ml_engine.get_uuid()))

        subprocess.Popen(
            uwsgi_start_cmd,
            shell=True,
        )
        self._logger.info("uwsgi was launched, waiting for it to run ...")
        self._monitor.verify_proper_startup()

        if not UwsgiConstants.DAEMONIZE:
            # The output should be read by a thread
            self._monitor.start()

        self._logger.info("'uwsgi' started successfully")

    #######################################################################
    # Methods that are accessed from uWSGI worker

    @classmethod
    def uwsgi_entry_point(
        cls,
        restful_comp,
        pipeline_name,
        model_path,
        deputy_id,
        stats_path_filename,
        within_uwsgi_context,
        standalone,
    ):
        cls._wid = 0 if not within_uwsgi_context else uwsgi.worker_id()

        cls.w_logger = logging.getLogger("{}.{}".format(cls.__module__, cls.__name__))

        cls.w_logger.info(
            "Entered to uWSGI entry point ... (wid: {}, pid: {}, ppid:{})".format(
                cls._wid, os.getpid(), os.getppid()
            )
        )

        cls.w_logger.info(
            "Restful comp (wid: {}, pid: {}, ppid:{}): {}".format(
                cls._wid, os.getpid(), os.getppid(), restful_comp
            )
        )

        cls.w_logger.info("Standalone mode (wid: {}): {}".format(cls._wid, standalone))
        restful_comp._ml_engine.set_standalone(standalone)
        cls._restful_comp = restful_comp
        cls._restful_comp._uuid_engine = deputy_id
        cls._restful_comp._stats_path_filename = stats_path_filename

        if within_uwsgi_context:
            UwsgiPostFork.init(cls)

        cls._setup_flask_app(pipeline_name, cls.w_logger)

        cls.register_model_load_handler(model_path, cls.w_logger, within_uwsgi_context)

        print("Deputy id {}".format(deputy_id))
        print("Stats path {}".format(stats_path_filename))
        # This should be the last printout in the uwsgi entry point function!!!
        print(WsgiMonitor.SUCCESS_RUN_INDICATION_MSG)

    @classmethod
    def restful_comp(cls):
        return cls._restful_comp

    @classmethod
    def _setup_flask_app(cls, pipeline_name, logger):
        app = FlaskAppWrapper(pipeline_name)

        for endpoint_entry in FlaskRoute.routes():
            logger.info("Add routing endpoint: {}".format(endpoint_entry))
            app.add_endpoint(
                url_rule=endpoint_entry[0],
                endpoint=endpoint_entry[1],
                handler=getattr(cls._restful_comp, endpoint_entry[2]),
                options=endpoint_entry[3],
                raw=endpoint_entry[4],
            )

        cls._application = FlaskAppWrapper.app
        cls._application.json_encoder = FlaskCustomJsonEncode
        cls._application.config["PROPAGATE_EXCEPTIONS"] = None

    @classmethod
    def register_model_load_handler(cls, model_path, logger, within_uwsgi_context):
        logger.info("Model file path: {}".format(model_path))

        if within_uwsgi_context:
            cls._model_selector = ModelSelector(
                model_path, cls._restful_comp._ml_engine.standalone
            )

            logger.info(
                "Register signal (model reloading): {} (worker, wid: {}, {})".format(
                    UwsgiConstants.MODEL_RELOAD_SIGNAL_NUM,
                    cls._wid,
                    UwsgiBroker._model_selector,
                )
            )
            uwsgi.register_signal(
                UwsgiConstants.MODEL_RELOAD_SIGNAL_NUM,
                "workers",
                cls._model_reload_signal,
            )

            logger.info(
                "Model path to monitor (signal {}, wid: {}): {}".format(
                    UwsgiConstants.MODEL_RELOAD_SIGNAL_NUM,
                    cls._wid,
                    cls._model_selector.model_env.sync_filepath,
                )
            )
            uwsgi.add_file_monitor(
                UwsgiConstants.MODEL_RELOAD_SIGNAL_NUM,
                cls._model_selector.model_env.sync_filepath,
            )
        else:
            if os.path.isfile(model_path):
                UwsgiBroker._restful_comp.load_model_callback(
                    model_path, stream=None, version=None
                )

    @staticmethod
    def _model_reload_signal(num):
        UwsgiBroker.w_logger.info(
            "Received model reload signal ... wid: {}, pid: {}".format(
                UwsgiBroker._wid, os.getpid()
            )
        )
        UwsgiBroker._reload_last_approved_model()

    @staticmethod
    def _reload_last_approved_model():
        if not UwsgiBroker._restful_comp or not UwsgiBroker._model_selector:
            raise MLPiperException(
                "Unexpected RESTful comp invariants! _restful_comp={}, _model_selector={}".format(
                    UwsgiBroker._restful_comp, UwsgiBroker._model_selector
                )
            )

        model_filepath = UwsgiBroker._model_selector.pick_model_filepath()
        UwsgiBroker.w_logger.debug(
            "Picking model (wid: {}, pid: {}, {}): {}".format(
                UwsgiBroker._wid,
                os.getpid(),
                UwsgiBroker._model_selector,
                model_filepath,
            )
        )

        # The following condition is here to handle an initial state, where a model was not set
        # for the given pipeline
        if model_filepath is not None and os.path.isfile(model_filepath):
            UwsgiBroker._restful_comp.load_model_callback(
                model_filepath, stream=None, version=None
            )
        else:
            UwsgiBroker.w_logger.info(
                "Model file does not exist: {}".format(model_filepath)
            )
