import glob
import logging
import os
import signal
import socket
import subprocess
import time
import atexit
import pandas as pd

from io import StringIO
from contextlib import closing

from datarobot_drum.drum.common import LOGGER_NAME_PREFIX, EnvVarNames
from mlpiper.components.connectable_component import ConnectableComponent

from py4j.java_gateway import GatewayParameters, CallbackServerParameters
from py4j.java_gateway import JavaGateway


class CustomScoringCode(ConnectableComponent):
    JAVA_COMPONENT_ENTRY_POINT_CLASS = "com.datarobot.custom.PredictorEntryPoint"
    JAVA_COMPONENT_CLASS_NAME = "com.datarobot.custom.ScoringCode"

    def __init__(self, engine):
        super(CustomScoringCode, self).__init__(engine)
        self.logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

        self.positive_class_label = None
        self.negative_class_label = None
        self.custom_model_path = None

        self._gateway = None
        self._predictor_via_py4j = None
        self._java_port = None
        self._atexit_cleanup = True
        self._proc = None
        # JVM the maximum heap size
        self._java_Xmx = os.environ.get(EnvVarNames.DRUM_JAVA_XMX)

        self._jar_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.jar"))

    def configure(self, params):
        super(CustomScoringCode, self).configure(params)
        self.custom_model_path = self._params["__custom_model_path__"]
        self.positive_class_label = self._params.get("positiveClassLabel")
        self.negative_class_label = self._params.get("negativeClassLabel")
        self._init_py4j_and_load_predictor()

        m = self._gateway.jvm.java.util.HashMap()
        for key in self._params.keys():
            if isinstance(self._params[key], dict):
                continue
            m[key] = self._params[key]
        self._predictor_via_py4j.configure(m)

    def _materialize(self, parent_data_objs, user_data):
        df = parent_data_objs[0]
        csv_data = self._convert_data_to_csv(df)
        out_csv = self._predictor_via_py4j.predict(csv_data)
        out_df = pd.read_csv(StringIO(out_csv))
        return [out_df]

    def _init_py4j_and_load_predictor(self):
        self._run_java_server_entry_point()
        self._setup_py4j_client_connection()
        if self._atexit_cleanup:
            atexit.register(self._cleanup)
        return self

    def _stop_py4j(self):
        """
        Stop the java process responsible for predictions
        """
        self._cleanup()

    def _run_java_server_entry_point(self):

        java_cp = ":".join(self._jar_files)

        self.logger.debug("java_cp: {}".format(java_cp))

        self._java_port = CustomScoringCode.find_free_port()
        cmd = ["java"]
        if self._java_Xmx:
            cmd.append("-Xmx{}".format(self._java_Xmx))
        cmd.extend(
            [
                "-cp",
                java_cp,
                CustomScoringCode.JAVA_COMPONENT_ENTRY_POINT_CLASS,
                "--class-name",
                CustomScoringCode.JAVA_COMPONENT_CLASS_NAME,
                "--port",
                str(self._java_port),
            ]
        )
        self.logger.debug("java gateway cmd: " + " ".join(cmd))

        self._proc = subprocess.Popen(
            cmd
        )  # , stdout=self._stdout_pipe_w, stderr=self._stderr_pipe_w)

        # TODO: provide a more robust way to check proper process startup
        time.sleep(2)

        poll_val = self._proc.poll()
        if poll_val is not None:
            raise Exception("java gateway failed to start")

        self.logger.debug("java server entry point run successfully!")

    def _setup_py4j_client_connection(self):
        gateway_params = GatewayParameters(
            port=self._java_port, auto_field=True, auto_close=True, eager_load=True
        )
        callback_server_params = CallbackServerParameters(
            port=0, daemonize=True, daemonize_connections=True, eager_load=True
        )
        self._gateway = JavaGateway(
            gateway_parameters=gateway_params,
            callback_server_parameters=callback_server_params,
            python_server_entry_point=self,
        )
        self._predictor_via_py4j = self._gateway.entry_point.getPredictor()
        if not self._predictor_via_py4j:
            raise Exception("None reference of py4j java object!")

        self.logger.debug(
            "Py4J component referenced successfully! comp_via_py4j: {}".format(
                self._predictor_via_py4j
            )
        )

    def _convert_data_to_csv(self, x):
        if isinstance(x, pd.DataFrame):
            self.logger.debug("Converting dataframe")
            return x.to_csv()
        else:
            print("[{}]".format(x))
            raise Exception("does not support other formats then dataframe: {}".format(type(x)))

    def _cleanup(self):
        """
        The cleanup function is called when the process exists
        """
        self.logger.debug("Cleaning py4j backend")

        if self._gateway:
            try:
                self.logger.debug("Shutting down py4j gateway ...")
                self._gateway.shutdown()
            except Exception as ex:
                self.logger.info("exception in gateway shutdown, {}".format(ex))

        if self._proc:
            self.logger.debug("Killing py4j gateway server ...")
            os.kill(self._proc.pid, signal.SIGTERM)
            os.kill(self._proc.pid, signal.SIGKILL)

    @staticmethod
    def find_free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
