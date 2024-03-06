"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import glob
import logging
import os
import py4j
import signal
import socket
import subprocess
import time
import atexit
import pandas as pd
import re
from itertools import chain
import tempfile

from io import StringIO
from contextlib import closing

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import RawPredictResponse
from datarobot_drum.drum.common import SupportedPayloadFormats
from datarobot_drum.drum.enum import (
    LOGGER_NAME_PREFIX,
    StructuredDtoKeys,
    UnstructuredDtoKeys,
    JavaArtifacts,
    EnvVarNames,
    PayloadFormat,
)
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor
from datarobot_drum.drum.exceptions import DrumCommonException

from py4j.java_gateway import GatewayParameters, CallbackServerParameters, JavaGateway
from py4j.java_collections import MapConverter
from werkzeug.datastructures import ImmutableMultiDict

RUNNING_LANG_MSG = "Running environment language: Java."


class JavaPredictor(BaseLanguagePredictor):
    JAVA_COMPONENT_ENTRY_POINT_CLASS = "com.datarobot.drum.PredictorEntryPoint"
    JAVA_COMPONENT_CLASS_NAME_DATAROBOT = "com.datarobot.drum.ScoringCode"
    JAVA_COMPONENT_CLASS_NAME_H2O = "com.datarobot.drum.H2OPredictor"
    JAVA_COMPONENT_CLASS_NAME_H2O_PIPELINE = "com.datarobot.drum.H2OPredictorPipeline"

    java_class_by_ext = {
        JavaArtifacts.JAR_EXTENSION: JAVA_COMPONENT_CLASS_NAME_DATAROBOT,
        JavaArtifacts.POJO_EXTENSION: JAVA_COMPONENT_CLASS_NAME_H2O,
        JavaArtifacts.MOJO_EXTENSION: JAVA_COMPONENT_CLASS_NAME_H2O,
        JavaArtifacts.MOJO_PIPELINE_EXTENSION: JAVA_COMPONENT_CLASS_NAME_H2O_PIPELINE,
    }

    def __init__(self):
        super(JavaPredictor, self).__init__()
        self.logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

        self._gateway = None
        self._predictor_via_py4j = None
        self._java_port = None
        self._atexit_cleanup = True
        self._proc = None
        # JVM the maximum heap size
        self._java_Xmx = os.environ.get(EnvVarNames.DRUM_JAVA_XMX)
        self._custom_predictor_class = os.environ.get(EnvVarNames.DRUM_JAVA_CUSTOM_PREDICTOR_CLASS)

        # init with only one system file `drum-py4j-entrypoint*.jar`
        # don't include default predictors it may cause deps conflict
        self._jar_files = glob.glob(
            os.path.join(os.path.dirname(__file__), "drum-py4j-entrypoint*.jar")
        )

    def mlpiper_configure(self, params):
        super(JavaPredictor, self).mlpiper_configure(params)

        # retrieve the relevant extensions of the java predictor
        # changed from last version significantly due to associating
        # jars with dr codegen AND h2o dai mojo pipeline
        self.custom_model_path = params["__custom_model_path__"]
        files_list = sorted(os.listdir(self.custom_model_path))
        files_list_str = " | ".join(files_list).lower()

        self.logger.debug("files in custom model path: ".format(files_list_str))
        reg_exp = r"|".join(r"(\{})".format(ext) for ext in JavaArtifacts.ALL)
        ext_re = re.findall(reg_exp, files_list_str)
        ext_re = [[match for match in matches if match != ""] for matches in ext_re]
        ext_re = list(chain.from_iterable(ext_re))

        # Note: files_list_str brought to lower case thus all the ext_re values are in lower case.
        if len(ext_re) == 0:
            raise DrumCommonException(
                "\n\n{}\n"
                "Could not find model artifact file in: {} supported by default predictors.\n"
                "They support filenames with the following extensions {}.\n"
                "List of retrieved files are: {}".format(
                    RUNNING_LANG_MSG, self.custom_model_path, JavaArtifacts.ALL, files_list_str
                )
            )
        self.logger.debug("relevant artifact extensions {}".format(", ".join(ext_re)))

        if JavaArtifacts.MOJO_PIPELINE_EXTENSION in ext_re:
            ## check for liscense
            license_location = os.path.join(params["__custom_model_path__"], "license.sig")
            self.logger.debug("license location: {}".format(license_location))
            try:
                os.environ["DRIVERLESS_AI_LICENSE_FILE"]
            except:
                try:
                    os.environ["DRIVERLESS_AI_LICENSE_KEY"]
                except:
                    if not os.path.exists(license_location):
                        raise DrumCommonException(
                            "Cannot find license file for DAI Mojo Pipeline.\n"
                            "Make sure you have done one of the following:\n"
                            "\t* provided license.sig file in the artifacts\n"
                            "\t* set the environment variable DRIVERLESS_AI_LICENSE_FILE : A location of file with a license\n"
                            "\t* set the environment variable DRIVERLESS_AI_LICENSE_KEY : A license key"
                        )
                    else:
                        os.environ["DRIVERLESS_AI_LICENSE_FILE"] = license_location
            self.model_artifact_extension = JavaArtifacts.MOJO_PIPELINE_EXTENSION
        else:
            self.model_artifact_extension = ext_re[0]

        self.logger.debug("model artifact extension: {}".format(self.model_artifact_extension))

        ## only needed to add mojo runtime jars
        additional_jars = (
            None
            if self.model_artifact_extension != JavaArtifacts.MOJO_PIPELINE_EXTENSION
            else glob.glob(os.path.join(self.custom_model_path, "*.jar"))
        )

        ## the mojo runtime jars must be added to self.__jar_files to be passed to gateway
        try:
            self._jar_files.extend(additional_jars)
        except:
            pass
        ##

        self._init_py4j_and_load_predictor()

        m = self._gateway.jvm.java.util.HashMap()
        for key in params.keys():
            if isinstance(params[key], dict):
                continue
            elif isinstance(params[key], list):
                pylist = params[key]
                jarray = self._gateway.new_array(self._gateway.jvm.java.lang.String, len(pylist))
                for i, val in enumerate(pylist):
                    jarray[i] = str(val)
                m[key] = jarray
            else:
                m[key] = params[key]
        self._predictor_via_py4j.configure(m)

    @property
    def supported_payload_formats(self):
        formats = SupportedPayloadFormats()
        formats.add(PayloadFormat.CSV)
        return formats

    def has_read_input_data_hook(self):
        return False

    def _predict(self, **kwargs) -> RawPredictResponse:
        input_text_bytes = kwargs.get(StructuredDtoKeys.BINARY_DATA)

        # If data size is more than 33K, pass it as a file to Java,
        # as passing big chunks to py4j as an array is 10% slower
        DATA_BUFFER_LIMIT_33K = 33792
        if len(input_text_bytes) > DATA_BUFFER_LIMIT_33K:
            with tempfile.NamedTemporaryFile(mode="wb") as tf:
                tf.write(input_text_bytes)
                tf.flush()
                out_csv = self._predictor_via_py4j.predictCSV(tf.name)
        else:
            out_csv = self._predictor_via_py4j.predict(input_text_bytes)

        out_df = pd.read_csv(StringIO(out_csv))
        return RawPredictResponse(out_df.values, out_df.columns)

    def predict_unstructured(self, data, **kwargs):
        mimetype = kwargs.get(UnstructuredDtoKeys.MIMETYPE, "")
        query = kwargs.get(UnstructuredDtoKeys.QUERY, dict())
        charset = kwargs.get(UnstructuredDtoKeys.CHARSET, "utf-8")
        if isinstance(data, bytes):
            ba = bytearray(data)
        elif isinstance(data, str):
            ba = bytearray(data.encode(charset))
        else:
            raise DrumCommonException("data of type {type(data)} is not supported")
        if isinstance(query, dict):
            query_dict = query
        elif isinstance(query, ImmutableMultiDict):
            query_dict = query.to_dict()
        jmap = MapConverter().convert(query_dict, self._gateway._gateway_client)
        ret = self._predictor_via_py4j.predictUnstructured(ba, mimetype, charset, jmap)
        if isinstance(ret, (str, bytes, type(None))):
            ret = ret, None
        elif isinstance(ret, bytearray):
            ret = bytes(ret), None
        elif isinstance(ret, tuple):
            ret = ret
        return ret

    def _transform(self, **kwargs):
        raise DrumCommonException("Transform feature is not supported for Java/Scala")

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

    def _run_java_server_entry_point(self, poll_timeout: int = 2) -> None:
        """
        Run the py4j gateway implemented in the Java predictor

        Parameters:
        -----------

        poll_timeout:
            timeout before checking whether gateway has been successfully started.
        """
        custom_class_path = os.environ.get(EnvVarNames.DRUM_JAVA_CUSTOM_CLASS_PATH)
        if custom_class_path:
            self.logger.debug("Custom class path: {}".format(custom_class_path))
            self._jar_files.append(custom_class_path)
        else:
            # find DRUM system file `predictors.jar` next to the current file in its installation dir
            _drum_default_predictors_jar = glob.glob(
                os.path.join(os.path.dirname(__file__), "drum-predictors-*.jar")
            )
            self._jar_files.extend(_drum_default_predictors_jar)

        java_cp = ":".join(self._jar_files)

        self.logger.debug("Full Java class path: {}".format(java_cp))

        self._java_port = JavaPredictor.find_free_port()

        cmd = ["java"]
        if self._java_Xmx:
            cmd.append("-Xmx{}".format(self._java_Xmx))

        class_to_load = (
            self._custom_predictor_class
            or JavaPredictor.java_class_by_ext[self.model_artifact_extension]
        )
        self.logger.info("Loading predictor class: {}".format(class_to_load))

        PY_TO_LOG4J2_LEVELS = {
            "NOTSET": "ALL",
            "DEBUG": "DEBUG",
            "INFO": "INFO",
            "WARNING": "WARN",
            "ERROR": "ERROR",
            "CRITICAL": "FATAL",
        }
        cmd.extend(
            [
                "-Dorg.apache.logging.log4j.level={}".format(
                    PY_TO_LOG4J2_LEVELS.get(
                        logging.getLevelName(self.logger.getEffectiveLevel()), "INFO"
                    )
                ),
                "-cp",
                java_cp,
                JavaPredictor.JAVA_COMPONENT_ENTRY_POINT_CLASS,
                "--class-name",
                class_to_load,
                "--port",
                str(self._java_port),
            ]
        )
        self.logger.debug("java gateway cmd: " + " ".join(cmd))

        self._proc = subprocess.Popen(
            cmd
        )  # , stdout=self._stdout_pipe_w, stderr=self._stderr_pipe_w)

        # TODO: provide a more robust way to check proper process startup
        time.sleep(poll_timeout)

        poll_val = self._proc.poll()
        if poll_val is not None:
            stdo, stde = self._proc.communicate()
            if stdo is not None:
                print(stdo.decode())
            if stde is not None:
                print(stde.decode())
            raise DrumCommonException("java gateway failed to start")

        self.logger.debug("java server entry point run successfully!")

    def _setup_py4j_client_connection(self):
        gateway_params = GatewayParameters(
            port=self._java_port, auto_field=True, auto_close=True, eager_load=True
        )

        callback_server_params = CallbackServerParameters(
            port=0, daemonize=True, daemonize_connections=True, eager_load=True
        )

        retries = 10
        while True:
            try:
                self._gateway = JavaGateway(
                    gateway_parameters=gateway_params,
                    callback_server_parameters=callback_server_params,
                    python_server_entry_point=self,
                )
                break
            except py4j.java_gateway.Py4JNetworkError as e:
                retries -= 1
                if retries <= 0:
                    break
                time.sleep(1)

        if self._gateway is None:
            self.logger.error("Failed to connect to java gateway")
            raise DrumCommonException("Failed to connect to java gateway")

        self.logger.debug("java server entry point run successfully!")

        self._predictor_via_py4j = self._gateway.entry_point.getPredictor()
        if not self._predictor_via_py4j:
            raise Exception("None reference of py4j java object!")

        self.logger.debug(
            "Py4J component referenced successfully! comp_via_py4j: {}".format(
                self._predictor_via_py4j
            )
        )

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
