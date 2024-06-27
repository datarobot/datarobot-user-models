import errno
import tempfile

import pytest
import uuid
import os
import logging
import json
import subprocess
import warnings

from mlpiper.ml_engine.sagemaker_engine import SageMakerEngine
from mlpiper.pipeline.components_desc import ComponentsDesc
from mlpiper.ml_engine.python_engine import PythonEngine
from mlpiper.ml_engine.py_spark_engine import PySparkEngine
from mlpiper.pipeline.dag import Dag
from mlpiper.pipeline.executor import Executor, ExecutorException
from mlpiper.pipeline.executor_config import ExecutorConfig

import mlpiper.pipeline.json_fields as json_fields

from .constants import COMPONENTS_ROOT
from .constants import GENERIC_COMPONENTS_ROOT
from .constants import JAVA_COMPONENTS_PATH
from .constants import MLPIPER_JAR_FILEPATH
from .constants import PYTHON_COMPONENTS_PATH
from .constants import R_COMPONENTS_PATH


class TestPythonEngine:

    system_config = {
        "statsDBHost": "localhost",
        "statsDBPort": 8086,
        "mlObjectSocketSinkPort": 7777,
        "mlObjectSocketSourcePort": 1,
        "workflowInstanceId": "8117aced55d7427e8cb3d9b82e4e26ac",
        "statsMeasurementID": "1",
        "modelFileSourcePath": "__fill_in_real_model__",
    }

    @staticmethod
    def _get_mlpiper_jar():
        if not os.path.exists(MLPIPER_JAR_FILEPATH):
            raise Exception("File: {} does not exists".format(MLPIPER_JAR_FILEPATH))
        return MLPIPER_JAR_FILEPATH

    @staticmethod
    def _gen_model_file():
        model_file = os.path.join("/tmp", "model_file_" + str(uuid.uuid4()))
        with open(model_file, "w") as ff:
            ff.write("model-1234")
        return model_file

    @staticmethod
    def _fix_pipeline(pipeline, model_file, **kwargs):
        system_config = TestPythonEngine.system_config
        system_config["modelFileSourcePath"] = model_file
        for key, value in kwargs.items():
            system_config[key] = value
        pipeline["systemConfig"] = system_config

    def _get_executor_config(self, pipeline, comp_root_path=PYTHON_COMPONENTS_PATH):
        config = ExecutorConfig(
            pipeline=json.dumps(pipeline),
            pipeline_file=None,
            run_locally=False,
            comp_root_path=comp_root_path,
            mlpiper_jar=None,
            spark_jars=None,
        )
        return config

    # Note, skip lines are commented so can be easily uncommented when debugging
    def test_dag_detect_is_stand_alone(self):

        pipeline = {
            "name": "stand_alone_test",
            "engineType": "Generic",
            "pipe": [
                {
                    "name": "Hello",
                    "id": 1,
                    "type": "hello-world",
                    "parents": [],
                    "arguments": {"arg1": "arg1-value"},
                }
            ],
        }
        python_engine = PythonEngine("test-pipe")
        comps_desc_list = ComponentsDesc(
            python_engine, pipeline=pipeline, comp_root_path=PYTHON_COMPONENTS_PATH
        ).load()
        dag = Dag(pipeline, comps_desc_list, python_engine)
        assert dag.is_stand_alone is True

    # @pytest.mark.skip(reason="skipping this test for now - debugging")
    def test_execute_python_stand_alone(self):
        pipeline = {
            "name": "stand_alone_test",
            "engineType": "Generic",
            "pipe": [
                {
                    "name": "Hello",
                    "id": 1,
                    "type": "hello-world",
                    "parents": [],
                    "arguments": {"arg1": "arg1-value"},
                }
            ],
        }
        self._fix_pipeline(pipeline, None)
        config = self._get_executor_config(pipeline)
        Executor(config).go()

    # @pytest.mark.skip(reason="skipping this test for now - debugging")
    def test_execute_python_stand_alone_with_argument_from_env_var(self):
        pipeline = {
            "name": "stand_alone_test",
            "engineType": "Generic",
            "pipe": [
                {
                    "name": "Hello",
                    "id": 1,
                    "type": "test-argument-from-env-var",
                    "parents": [],
                    "arguments": {"arg1": "test-value", "fromEnvVar2": "test-value2"},
                }
            ],
        }
        self._fix_pipeline(pipeline, None)
        config = self._get_executor_config(pipeline)
        os.environ.setdefault("TEST_VAR", "test-value")
        os.environ.setdefault("TEST_VAR2", "non test value")
        Executor(config).go()

    # @pytest.mark.skip(reason="skipping this test for now - debugging")
    def test_execute_python_stand_alone_with_exit_0(self):
        pipeline = {
            "name": "stand_alone_test",
            "engineType": "Generic",
            "pipe": [
                {
                    "name": "Hello",
                    "id": 1,
                    "type": "test-argument-from-env-var",
                    "parents": [],
                    "arguments": {"arg1": "test-exit-0", "fromEnvVar2": "test-value2"},
                }
            ],
        }
        self._fix_pipeline(pipeline, None)
        config = self._get_executor_config(pipeline)
        Executor(config).go()

    # @pytest.mark.skip(reason="skipping this test for now - debugging")
    def test_execute_python_stand_alone_with_exit_1(self):
        pipeline = {
            "name": "stand_alone_test",
            "engineType": "Generic",
            "pipe": [
                {
                    "name": "Hello",
                    "id": 1,
                    "type": "test-argument-from-env-var",
                    "parents": [],
                    "arguments": {"arg1": "test-exit-1", "fromEnvVar2": "test-value2"},
                }
            ],
        }
        self._fix_pipeline(pipeline, None)
        config = self._get_executor_config(pipeline)
        passed = 0
        try:
            Executor(config).go()
        except ExecutorException as e:
            passed = str(e).startswith("Pipeline called exit(), with code: 1")
        assert passed

    # @pytest.mark.skip(reason="skipping this test for now - debugging")
    def test_execute_python_connected(self, caplog):
        pipeline = {
            "name": "stand_alone_test",
            "engineType": "Generic",
            "pipe": [
                {
                    "name": "src",
                    "id": 1,
                    "type": "string-source",
                    "parents": [],
                    "arguments": {"value": "test-st1-1234"},
                },
                {
                    "name": "sink",
                    "id": 2,
                    "type": "string-sink",
                    "parents": [{"parent": 1, "output": 0}],
                    "arguments": {"expected-value": "test-st1-1234"},
                },
            ],
        }
        self._fix_pipeline(pipeline, None)
        config = self._get_executor_config(pipeline)
        Executor(config).go()

    def test_execute_python_connected_test_mode(self, caplog):
        pipeline = {
            "name": "stand_alone_test",
            "engineType": "Generic",
            "pipe": [
                {
                    "name": "src",
                    "id": 1,
                    "type": "string-source",
                    "parents": [],
                    "arguments": {"value": "test-st1-1234", "check-test-mode": True},
                },
                {
                    "name": "sink",
                    "id": 2,
                    "type": "string-sink",
                    "parents": [{"parent": 1, "output": 0}],
                    "arguments": {
                        "expected-value": "test-st1-1234",
                        "check-test-mode": True,
                    },
                },
            ],
        }
        self._fix_pipeline(pipeline, None, __test_mode__=True)
        config = self._get_executor_config(pipeline)
        Executor(config).go()

    @staticmethod
    def _R_tool_installed():
        try:
            subprocess.call(["Rscript", "--version"])
            return True
        except OSError as e:
            if e.errno == errno.ENOENT:
                return False
            raise e

    @pytest.mark.skipif(
        condition="JARVIS_VERSION" in os.environ,
        reason="Skip 'R' steps until it'll be possible to install 'R' dependencies in Jarvis",
    )
    def test_execute_r_stand_alone(self):
        if not TestPythonEngine._R_tool_installed():
            warnings.warn("WARNING: Rscript is not installed. Skipping unit-test.")
            return

        pipeline = {
            "name": "stand_alone_test",
            "engineType": "Generic",
            "pipe": [
                {
                    "name": "R",
                    "id": 1,
                    "type": "test-r-predict",
                    "parents": [],
                    "arguments": {"data-file": "/tmp/ddd.data"},
                }
            ],
        }

        model_file = self._gen_model_file()
        self._fix_pipeline(pipeline, model_file)
        try:
            config = self._get_executor_config(pipeline, R_COMPONENTS_PATH)
            Executor(config).go()
        finally:
            os.remove(model_file)

    @pytest.mark.skipif(
        condition="JARVIS_VERSION" in os.environ,
        reason="Skip 'R' steps until it'll be possible to install 'R' dependencies in Jarvis",
    )
    def test_execute_r_connected(self):
        if not TestPythonEngine._R_tool_installed():
            warnings.warn("WARNING: Rscript is not installed. Skipping unit-test.")
            return

        pipeline = {
            "name": "connected_java_test",
            "engineType": "Generic",
            "pipe": [
                {
                    "name": "src",
                    "id": 1,
                    "type": "string-source",
                    "parents": [],
                    "arguments": {"value": "test-st1-1234"},
                },
                {
                    "name": "infer",
                    "id": 2,
                    "type": "test-r-predict-middle",
                    "parents": [{"parent": 1, "output": 0}],
                    "arguments": {"iter": 1, "expected_input_str": "test-st1-1234"},
                },
                {
                    "name": "sink",
                    "id": 3,
                    "type": "string-sink",
                    "parents": [{"parent": 2, "output": 0}],
                    "arguments": {"expected-value": "test-st1-1234"},
                },
            ],
        }

        model_file = self._gen_model_file()
        self._fix_pipeline(pipeline, model_file)
        try:
            config = self._get_executor_config(pipeline, GENERIC_COMPONENTS_ROOT)
            Executor(config).go()
        finally:
            os.remove(model_file)

    # @pytest.mark.skip(reason="skipping this test for now - debugging")
    def test_python_stand_alone_argument_building(self):
        systemConfig = {
            "statsDBHost": "localhost",
            "statsDBPort": 8899,
            "statsMeasurementID": "tf-job-0001",
            "mlObjectSocketHost": "localhost",
            "mlObjectSocketSourcePort": 9900,
            "mlObjectSocketSinkPort": 9901,
            "modelFileSinkPath": "output-model-1234",
            "modelFileSourcePath": "input-model-1234",
            "healthStatFilePath": "/tmp/health",
            "workflowInstanceId": "/tmp/run/filesink1",
            "socketSourcePort": 0,
            "socketSinkPort": 0,
            "enableHealth": True,
            "canaryThreshold": 0.0,
        }
        ee_config = {}
        pipeline = {
            "name": "stand_alone_test",
            "engineType": "Generic",
            "pipe": [
                {
                    "name": "Test Train",
                    "id": 1,
                    "type": "test-python-train",
                    "parents": [],
                    "arguments": {"arg1": "arg1-value"},
                }
            ],
        }
        python_engine = PythonEngine("test-pipe")
        comps_desc_list = ComponentsDesc(
            python_engine, pipeline=pipeline, comp_root_path=PYTHON_COMPONENTS_PATH
        ).load()
        dag = Dag(pipeline, comps_desc_list, python_engine)

        dag_node = dag.get_dag_node(0)
        input_args = dag_node.input_arguments(
            systemConfig, ee_config, comp_only_args=True
        )
        assert input_args["arg1"] == "arg1-value"
        assert input_args["output-model"] == "output-model-1234"


    def test_pipeline_component_termination(self):
        with tempfile.NamedTemporaryFile() as f:
            pipeline = {
                "name": "pipeline_component_termination_test",
                "engineType": "Generic",
                "pipe": [
                    {
                        "name": "Dummy Component To Test Termination",
                        "id": 1,
                        "type": "dummy-component-to-test-termination",
                        "parents": [],
                        "arguments": {
                            "filepath-for-termination-output": f.name
                        },
                    }
                ],
            }
            config = self._get_executor_config(pipeline)
            Executor(config).go()

            assert f.read().decode('utf-8') == 'Termination was handled successfully'
