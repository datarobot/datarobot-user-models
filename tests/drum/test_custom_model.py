import collections
import glob
import json
import os
import pickle
import re
import shutil
import signal
import subprocess
import time
from tempfile import mkdtemp, NamedTemporaryFile, TemporaryDirectory
from threading import Thread
from unittest import mock
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import requests

from datarobot_drum.drum.common import (
    CUSTOM_FILE_NAME,
    CustomHooks,
    ArgumentsOptions,
    PythonArtifacts,
)

from datarobot_drum.drum.runtime import DrumRuntime

# Framweork keywords
XGB = "xgboost"
XGB_INFERENCE = "xgb_inference"
XGB_TRAINING = "xgb_training"

KERAS = "keras"
KERAS_INFERENCE = "keras_inference"
KERAS_INFERENCE_JOBLIB = "keras_inference_joblib"
KERAS_TRAINING_JOBLIB = "keras_training_joblib"

SKLEARN = "sklearn"
SKLEARN_TRAINING = "sklearn_training"
SKLEARN_INFERENCE = "sklearn_inference"

PYTORCH = "pytorch"
PYTORCH_INFERENCE = "pytorch_inference"

PYPMML = "pypmml"

RDS = "rds"
CODEGEN = "jar"
MULTI_ARTIFACT = "multiartifact"

# Problem keywords, used to mark datasets
REGRESSION = "regression"
REGRESSION_INFERENCE = "regression_inference"
BINARY = "binary"

# Language keywords
PYTHON = "python3"
NO_CUSTOM = "no_custom"
PYTHON_ALL_HOOKS = "python_all_hooks"
PYTHON_LOAD_MODEL = "python_load_model"
R = "R"
R_ALL_HOOKS = "R_all_hooks"
JAVA = "java"
PYTHON_XGBOOST_CLASS_LABELS_VALIDATION = "predictions_and_class_labels_validation"

DOCKER_PYTHON_SKLEARN = "cmrunner_test_env_python_sklearn"

RESPONSE_PREDICTIONS_KEY = "predictions"

WEIGHTS_ARGS = "weights-args"
WEIGHTS_CSV = "weights-csv"


class DrumServerProcess:
    def __init__(self):
        self.process = None
        self.out_stream = None
        self.err_stream = None

    @property
    def returncode(self):
        return self.process.returncode


class DrumServerRun:
    def __init__(
        self, framework, problem, language, docker=None, custom_model_dir=None, force_start=False,
    ):
        self._custom_model_dir_to_remove = None
        if custom_model_dir is None:
            # If custom model directory is not set explicitly - it's is created for the class.
            # In such case it should also be removed.
            custom_model_dir = TestCMRunner._create_custom_model_dir(framework, problem, language)
            self._custom_model_dir_to_remove = custom_model_dir

        port = 6799
        server_address = "localhost:{}".format(port)
        url_host = os.environ.get("TEST_URL_HOST", "localhost")
        if docker:
            self.url_server_address = "http://{}:{}".format(url_host, port)
        else:
            self.url_server_address = "http://localhost:{}".format(port)

        cmd = "{} server --code-dir {} --address {}".format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, server_address
        )
        cmd = TestCMRunner._cmd_add_class_labels(cmd, framework, problem)
        if docker:
            cmd += " --docker {}".format(docker)
        if force_start:
            cmd += " --force-start-internal"
        self._cmd = cmd

        self._process_object_holder = DrumServerProcess()
        self._server_thread = None

    def __enter__(self):
        self._server_thread = Thread(
            target=TestCMRunner.run_server_thread, args=(self._cmd, self._process_object_holder)
        )
        self._server_thread.start()
        time.sleep(0.5)

        TestCMRunner.wait_for_server(
            self.url_server_address, timeout=10, process_holder=self._process_object_holder
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # shutdown server
        response = requests.post(self.url_server_address + "/shutdown/")
        assert response.ok
        time.sleep(1)

        self._server_thread.join()

        if self._custom_model_dir_to_remove is not None:
            TestCMRunner._delete_custom_model_dir(self._custom_model_dir_to_remove)

    @property
    def process(self):
        return self._process_object_holder or None


class TestCMRunner:
    @classmethod
    def setup_class(cls):
        cls.tests_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.tests_fixtures_path = os.path.join(cls.tests_root_path, "fixtures")
        cls.tests_artifacts_path = os.path.join(cls.tests_fixtures_path, "drop_in_model_artifacts")
        cls.tests_data_path = os.path.join(cls.tests_root_path, "testdata")
        cls.model_templates_path = os.path.join(cls.tests_root_path, "..", "model_templates")

        cls.paths_to_real_models = {
            (PYTHON, SKLEARN): os.path.join(cls.model_templates_path, "python3_sklearn"),
            (PYTHON, KERAS_INFERENCE): os.path.join(
                cls.model_templates_path, "inference/python3_keras"
            ),
            (PYTHON, KERAS_INFERENCE_JOBLIB): os.path.join(
                cls.model_templates_path, "inference/python3_keras_joblib"
            ),
            (PYTHON, KERAS_TRAINING_JOBLIB): os.path.join(
                cls.model_templates_path, "training/python3_keras_joblib"
            ),
            (PYTHON, XGB_INFERENCE): os.path.join(
                cls.model_templates_path, "inference/python3_xgboost"
            ),
            (PYTHON, XGB_TRAINING): os.path.join(
                cls.model_templates_path, "training/python3_xgboost"
            ),
            (PYTHON, SKLEARN_INFERENCE): os.path.join(
                cls.model_templates_path, "inference/python3_sklearn"
            ),
            (PYTHON, SKLEARN_TRAINING): os.path.join(
                cls.model_templates_path, "training/python3_sklearn"
            ),
            (PYTHON, PYTORCH_INFERENCE): os.path.join(
                cls.model_templates_path, "inference/python3_pytorch"
            ),
        }

        cls.fixtures = {
            PYTHON: (os.path.join(cls.tests_fixtures_path, "custom.py"), "custom.py"),
            NO_CUSTOM: (None, None),
            PYTHON_ALL_HOOKS: (
                os.path.join(cls.tests_fixtures_path, "all_hooks_custom.py"),
                "custom.py",
            ),
            PYTHON_XGBOOST_CLASS_LABELS_VALIDATION: (
                os.path.join(cls.tests_fixtures_path, "pred_validation_custom.py"),
                "custom.py",
            ),
            PYTHON_LOAD_MODEL: (
                os.path.join(cls.tests_fixtures_path, "load_model_custom.py"),
                "custom.py",
            ),
            R: (os.path.join(cls.tests_fixtures_path, "custom.R"), "custom.R"),
            R_ALL_HOOKS: (os.path.join(cls.tests_fixtures_path, "all_hooks_custom.R"), "custom.R"),
        }
        cls.datasets = {
            # If specific dataset should be defined for a framework, use (framework, problem) key.
            # Otherwise default dataset is used (None, problem)
            (None, REGRESSION): os.path.join(cls.tests_data_path, "boston_housing.csv"),
            (PYPMML, REGRESSION): os.path.join(cls.tests_data_path, "iris_binary_training.csv"),
            (None, REGRESSION_INFERENCE): os.path.join(
                cls.tests_data_path, "boston_housing_inference.csv"
            ),
            (None, BINARY): os.path.join(cls.tests_data_path, "iris_binary_training.csv"),
        }

        cls.artifacts = {
            (None, REGRESSION): None,
            (None, BINARY): None,
            (SKLEARN, REGRESSION): os.path.join(cls.tests_artifacts_path, "sklearn_reg.pkl"),
            (SKLEARN, REGRESSION_INFERENCE): os.path.join(
                cls.tests_artifacts_path, "sklearn_reg.pkl"
            ),
            (MULTI_ARTIFACT, REGRESSION): [
                os.path.join(cls.tests_artifacts_path, "sklearn_reg.pkl"),
                os.path.join(cls.tests_artifacts_path, "keras_reg.h5"),
            ],
            (SKLEARN, BINARY): os.path.join(cls.tests_artifacts_path, "sklearn_bin.pkl"),
            (KERAS, REGRESSION): os.path.join(cls.tests_artifacts_path, "keras_reg.h5"),
            (KERAS, BINARY): os.path.join(cls.tests_artifacts_path, "keras_bin.h5"),
            (XGB_INFERENCE, REGRESSION): os.path.join(cls.tests_artifacts_path, "xgb_reg.pkl"),
            (XGB_INFERENCE, BINARY): os.path.join(cls.tests_artifacts_path, "xgb_bin.pkl"),
            (PYTORCH, REGRESSION): [
                os.path.join(cls.tests_artifacts_path, "torch_reg.pth"),
                os.path.join(cls.tests_artifacts_path, "PyTorch.py"),
            ],
            (PYTORCH, BINARY): [
                os.path.join(cls.tests_artifacts_path, "torch_bin.pth"),
                os.path.join(cls.tests_artifacts_path, "PyTorch.py"),
            ],
            (RDS, REGRESSION): os.path.join(cls.tests_artifacts_path, "r_reg.rds"),
            (RDS, BINARY): os.path.join(cls.tests_artifacts_path, "r_bin.rds"),
            (CODEGEN, REGRESSION): os.path.join(cls.tests_artifacts_path, "java_reg.jar"),
            (CODEGEN, BINARY): os.path.join(cls.tests_artifacts_path, "java_bin.jar"),
            (PYPMML, REGRESSION): os.path.join(cls.tests_artifacts_path, "iris_reg.pmml"),
            (PYPMML, BINARY): os.path.join(cls.tests_artifacts_path, "iris_bin.pmml"),
        }

        cls.target = {BINARY: "Species", REGRESSION: "MEDV"}
        cls.class_labels = {
            (SKLEARN, BINARY): ["Iris-setosa", "Iris-versicolor"],
            (XGB, BINARY): ["Iris-setosa", "Iris-versicolor"],
            (KERAS, BINARY): ["Iris-setosa", "Iris-versicolor"],
            (RDS, BINARY): ["Iris-setosa", "Iris-versicolor"],
            (PYPMML, BINARY): ["Iris-setosa", "Iris-versicolor"],
        }

    @classmethod
    def teardown_class(cls):
        pass

    @staticmethod
    def _exec_shell_cmd(cmd, err_msg, assert_if_fail=True, process_obj_holder=None, env=os.environ):
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            env=env,
            universal_newlines=True,
        )
        if process_obj_holder is not None:
            process_obj_holder.process = p

        (stdout, stderr) = p.communicate()

        if process_obj_holder is not None:
            process_obj_holder.out_stream = stdout
            process_obj_holder.err_stream = stderr

        if p.returncode != 0:
            print("stdout: {}".format(stdout))
            print("stderr: {}".format(stderr))
            if assert_if_fail:
                assert p.returncode == 0, err_msg

        return p, stdout, stderr

    @classmethod
    def _create_custom_model_dir(cls, framework, problem, language, is_training=False):
        custom_model_dir = mkdtemp(prefix="custom_model_", dir="/tmp")
        if is_training:
            model_template_dir = cls._get_template_dir(language, framework, is_training)
            for filename in glob.glob(r"{}/*.py".format(model_template_dir)):
                shutil.copy2(filename, custom_model_dir)
        else:
            artifact_filenames = cls._get_artifact_filename(framework, problem)
            if artifact_filenames is not None:
                if not isinstance(artifact_filenames, list):
                    artifact_filenames = [artifact_filenames]
                for filename in artifact_filenames:
                    shutil.copy2(filename, custom_model_dir)

            fixture_filename, rename = cls._get_fixture_filename(language)
            if fixture_filename:
                shutil.copy2(fixture_filename, os.path.join(custom_model_dir, rename))
        return custom_model_dir

    @staticmethod
    def _delete_custom_model_dir(custom_model_dir):
        shutil.rmtree(custom_model_dir)

    @classmethod
    def _get_artifact_filename(cls, framework, problem):
        return cls.artifacts[(framework, problem)]

    @classmethod
    def _get_template_dir(cls, language, framework, is_training):
        if is_training:
            if framework == KERAS:
                return cls.paths_to_real_models[(language, KERAS_TRAINING_JOBLIB)]
            elif framework == XGB:
                return cls.paths_to_real_models[(language, XGB_TRAINING)]
            elif framework == SKLEARN:
                return cls.paths_to_real_models[(language, SKLEARN_TRAINING)]
        return cls.paths_to_real_models[(language, framework)]

    @classmethod
    def _get_class_labels(cls, framework, problem):
        return cls.class_labels.get((framework, problem), None)

    @classmethod
    def _get_dataset_filename(cls, framework, problem):
        framework_key = framework
        problem_key = problem
        # if specific dataset for framework was not defined,
        # use default dataset for this problem, e.g. (None, problem)
        framework_key = None if (framework_key, problem_key) not in cls.datasets else framework_key
        return cls.datasets[(framework_key, problem_key)]

    @classmethod
    def _get_fixture_filename(cls, language):
        return cls.fixtures[language]

    @classmethod
    def _cmd_add_class_labels(cls, cmd, framework, problem):
        if problem != BINARY:
            return cmd

        labels = cls._get_class_labels(framework, problem)
        pos = labels[1] if labels else "yes"
        neg = labels[0] if labels else "no"
        cmd = cmd + " --positive-class-label {} --negative-class-label {}".format(pos, neg)
        return cmd

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None),
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),
            (SKLEARN, BINARY, PYTHON, None),
            (KERAS, REGRESSION, PYTHON, None),
            (KERAS, BINARY, PYTHON, None),
            (XGB_INFERENCE, REGRESSION, PYTHON, None),
            (XGB_INFERENCE, BINARY, PYTHON, None),
            (XGB_INFERENCE, BINARY, PYTHON_XGBOOST_CLASS_LABELS_VALIDATION, None),
            (PYTORCH, REGRESSION, PYTHON, None),
            (PYTORCH, BINARY, PYTHON, None),
            (RDS, REGRESSION, R, None),
            (RDS, BINARY, R, None),
            (CODEGEN, REGRESSION, NO_CUSTOM, None),
            (CODEGEN, BINARY, NO_CUSTOM, None),
            (MULTI_ARTIFACT, REGRESSION, PYTHON_LOAD_MODEL, None),
            (PYPMML, REGRESSION, NO_CUSTOM, None),
            (PYPMML, BINARY, NO_CUSTOM, None),
        ],
    )
    def test_custom_models_with_drum(self, framework, problem, language, docker):
        custom_model_dir = self._create_custom_model_dir(framework, problem, language)

        input_dataset = self._get_dataset_filename(framework, problem)

        with NamedTemporaryFile() as output:
            cmd = "{} score --code-dir {} --input {} --output {}".format(
                ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset, output.name
            )
            cmd = self._cmd_add_class_labels(cmd, framework, problem)
            if docker:
                cmd += " --docker {} --verbose ".format(docker)

            TestCMRunner._exec_shell_cmd(
                cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
            )
            TestCMRunner._delete_custom_model_dir(custom_model_dir)
            in_data = pd.read_csv(input_dataset)
            out_data = pd.read_csv(output.name)
            assert in_data.shape[0] == out_data.shape[0]

    @pytest.mark.parametrize(
        "framework, problem, language", [(SKLEARN, BINARY, PYTHON), (RDS, BINARY, R)]
    )
    def test_bin_models_with_wrong_labels(self, framework, problem, language):
        custom_model_dir = self._create_custom_model_dir(framework, problem, language)

        input_dataset = self._get_dataset_filename(framework, problem)
        cmd = "{} score --code-dir {} --input {}".format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset
        )
        if problem == BINARY:
            cmd = cmd + " --positive-class-label yes --negative-class-label no"

        p, stdo, stde = TestCMRunner._exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )
        if framework == SKLEARN:
            assert (
                str(stde).find("Wrong class labels. Use class labels detected by sklearn model")
                != -1
            )
        elif framework == RDS:
            assert (
                str(stde).find("Wrong class labels. Use class labels according to your dataset")
                != -1
            )
        TestCMRunner._delete_custom_model_dir(custom_model_dir)

    # testing negative cases: no artifact, no custom;
    @pytest.mark.parametrize(
        "framework, problem, language",
        [
            (None, REGRESSION, NO_CUSTOM),  # no artifact, no custom
            (SKLEARN, REGRESSION, R),  # python artifact, custom.R
            (RDS, REGRESSION, PYTHON),  # R artifact, custom.py
            (None, REGRESSION, R),  # no artifact, custom.R without load_model
            (None, REGRESSION, PYTHON),  # no artifact, custom.py without load_model
        ],
    )
    def test_detect_language(self, framework, problem, language):
        custom_model_dir = self._create_custom_model_dir(framework, problem, language)

        input_dataset = self._get_dataset_filename(framework, problem)
        cmd = "{} score --code-dir {} --input {}".format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset
        )
        if problem == BINARY:
            cmd = cmd + " --positive-class-label yes --negative-class-label no"

        p, stdo, stde = TestCMRunner._exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )

        cases_1_2_3 = (
            str(stde).find("Can not detect language by artifacts and/or custom.py/R files") != -1
        )
        case_4 = (
            str(stde).find(
                "Could not find a serialized model artifact with .rds extension, supported by default R predictor. "
                "If your artifact is not supported by default predictor, implement custom.load_model hook."
            )
            != -1
        )
        case_5 = (
            str(stde).find(
                "Could not find model artifact file in: {} supported by default predictors".format(
                    custom_model_dir
                )
            )
            != -1
        )
        assert any([cases_1_2_3, case_4, case_5])

        TestCMRunner._delete_custom_model_dir(custom_model_dir)

    @pytest.mark.parametrize(
        "framework, language", [(SKLEARN, PYTHON_ALL_HOOKS), (RDS, R_ALL_HOOKS)]
    )
    def test_custom_model_with_all_predict_hooks(self, framework, language):
        custom_model_dir = self._create_custom_model_dir(framework, REGRESSION, language)
        input_dataset = self._get_dataset_filename(framework, REGRESSION)
        with NamedTemporaryFile() as output:
            cmd = "{} score --code-dir {} --input {} --output {}".format(
                ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset, output.name
            )
            TestCMRunner._exec_shell_cmd(
                cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
            )
            TestCMRunner._delete_custom_model_dir(custom_model_dir)
            preds = pd.read_csv(output.name)
            assert all(
                val for val in (preds["Predictions"] == len(CustomHooks.ALL_PREDICT)).values
            ), preds

    @staticmethod
    def run_server_thread(cmd, process_obj_holder):
        TestCMRunner._exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
            process_obj_holder=process_obj_holder,
        )

    @staticmethod
    def wait_for_server(url, timeout, process_holder):
        # waiting for ping to succeed
        while True:
            try:
                response = requests.get(url)
                if response.ok:
                    break
            except Exception:
                pass

            time.sleep(1)
            timeout -= 1
            if timeout <= 0:
                if process_holder is not None:
                    print("Killing subprocess: {}".format(process_holder.process.pid))
                    os.killpg(os.getpgid(process_holder.process.pid), signal.SIGTERM)
                    time.sleep(0.25)
                    os.killpg(os.getpgid(process_holder.process.pid), signal.SIGKILL)

                assert timeout, "Server failed to start: url: {}".format(url)

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),
            (SKLEARN, BINARY, PYTHON, None),
            (KERAS, REGRESSION, PYTHON, None),
            (KERAS, BINARY, PYTHON, None),
            (XGB_INFERENCE, REGRESSION, PYTHON, None),
            (XGB_INFERENCE, BINARY, PYTHON, None),
            (PYTORCH, REGRESSION, PYTHON, None),
            (PYTORCH, BINARY, PYTHON, None),
            (RDS, REGRESSION, R, None),
            (RDS, BINARY, R, None),
            (CODEGEN, REGRESSION, NO_CUSTOM, None),
            (CODEGEN, BINARY, NO_CUSTOM, None),
            (MULTI_ARTIFACT, REGRESSION, PYTHON_LOAD_MODEL, None),
            (PYPMML, REGRESSION, NO_CUSTOM, None),
            (PYPMML, BINARY, NO_CUSTOM, None),
        ],
    )
    def test_custom_models_with_drum_prediction_server(self, framework, problem, language, docker):
        with DrumServerRun(framework, problem, language, docker) as run:
            input_dataset = self._get_dataset_filename(framework, problem)

            # do predictions
            response = requests.post(
                run.url_server_address + "/predict/", files={"X": open(input_dataset)}
            )

            print(response.text)
            assert response.ok
            actual_num_predictions = len(json.loads(response.text)[RESPONSE_PREDICTIONS_KEY])
            in_data = pd.read_csv(input_dataset)
            assert in_data.shape[0] == actual_num_predictions

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [(SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN), (SKLEARN, BINARY, PYTHON, None)],
    )
    def test_custom_models_drum_prediction_server_response(
        self, framework, problem, language, docker
    ):
        with DrumServerRun(framework, problem, language, docker) as run:
            input_dataset = self._get_dataset_filename(framework, problem)

            # do predictions
            response = requests.post(
                run.url_server_address + "/predict/", files={"X": open(input_dataset)}
            )

            assert response.ok
            response_json = json.loads(response.text)
            assert isinstance(response_json, dict)
            assert RESPONSE_PREDICTIONS_KEY in response_json
            predictions_list = response_json[RESPONSE_PREDICTIONS_KEY]
            assert isinstance(predictions_list, list)
            assert len(predictions_list)
            prediction_item = predictions_list[0]
            if problem == BINARY:
                assert isinstance(prediction_item, dict)
                assert len(prediction_item) == 2
                assert all([isinstance(x, str) for x in prediction_item.keys()])
                assert all([isinstance(x, float) for x in prediction_item.values()])
            elif problem == REGRESSION:
                assert isinstance(prediction_item, float)

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [(SKLEARN, BINARY, PYTHON, None), (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN)],
    )
    def test_custom_models_perf_test(self, framework, problem, language, docker):
        custom_model_dir = self._create_custom_model_dir(framework, problem, language)

        input_dataset = self._get_dataset_filename(framework, problem)

        cmd = "{} perf-test -i 10 -s 1000 --code-dir {} --input {}".format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset
        )
        cmd = self._cmd_add_class_labels(cmd, framework, problem)
        if docker:
            cmd += " --docker {}".format(docker)

        TestCMRunner._exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
        TestCMRunner._delete_custom_model_dir(custom_model_dir)

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, BINARY, PYTHON, None),
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None),
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, DOCKER_PYTHON_SKLEARN),
        ],
    )
    def test_custom_models_validation_test(self, framework, problem, language, docker):
        custom_model_dir = self._create_custom_model_dir(framework, problem, language)

        input_dataset = self._get_dataset_filename(framework, problem)

        cmd = "{} validation --code-dir {} --input {}".format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset
        )
        cmd = self._cmd_add_class_labels(cmd, framework, problem)
        if docker:
            cmd += " --docker {}".format(docker)

        p, stdo, stde = TestCMRunner._exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )

        TestCMRunner._delete_custom_model_dir(custom_model_dir)
        if language == NO_CUSTOM:
            assert re.search(r"Null value imputation\s+FAILED", stdo)
        else:
            assert re.search(r"Null value imputation\s+PASSED", stdo)

    @pytest.mark.parametrize("language, language_suffix", [("python", ".py"), ("r", ".R")])
    def test_template_creation(self, language, language_suffix):
        print("Running template creation tests: {}".format(language))
        directory = os.path.join("/tmp", "template_test_{}".format(uuid4()))
        cmd = "{drum_prog} new model --language {language} --code-dir {directory}".format(
            drum_prog=ArgumentsOptions.MAIN_COMMAND, language=language, directory=directory
        )

        TestCMRunner._exec_shell_cmd(
            cmd, "Failed creating a template for custom model, cmd={}".format(cmd)
        )

        assert os.path.isdir(directory), "Directory {} does not exists (or not a dir)".format(
            directory
        )

        assert os.path.isfile(os.path.join(directory, "README.md"))
        custom_file = os.path.join(directory, CUSTOM_FILE_NAME + language_suffix)
        assert os.path.isfile(custom_file)
        TestCMRunner._delete_custom_model_dir(directory)

    @staticmethod
    def _add_weights_cmd(weights, input_csv):
        df = pd.read_csv(input_csv)
        colname = "some-colname"
        weights_data = pd.Series(np.random.randint(1, 3, len(df)))
        __keep_this_around = NamedTemporaryFile("w")
        if weights == WEIGHTS_ARGS:
            df[colname] = weights_data
            df.to_csv(__keep_this_around.name)
            return " --row-weights " + colname, __keep_this_around.name, __keep_this_around
        elif weights == WEIGHTS_CSV:
            weights_data.to_csv(__keep_this_around.name)
            return " --row-weights-csv " + __keep_this_around.name, input_csv, __keep_this_around

        return "", input_csv, __keep_this_around

    @pytest.mark.parametrize("framework", [SKLEARN, XGB, KERAS])
    @pytest.mark.parametrize("problem", [BINARY, REGRESSION])
    @pytest.mark.parametrize("language", [PYTHON])
    @pytest.mark.parametrize("docker", [DOCKER_PYTHON_SKLEARN, None])
    @pytest.mark.parametrize("weights", [WEIGHTS_CSV, WEIGHTS_ARGS, None])
    @pytest.mark.parametrize("use_output", [True, False])
    def test_fit(self, framework, problem, language, docker, weights, use_output):
        custom_model_dir = self._create_custom_model_dir(
            framework, problem, language, is_training=True
        )

        input_dataset = self._get_dataset_filename(framework, problem)

        weights_cmd, input_dataset, __keep_this_around = self._add_weights_cmd(
            weights, input_dataset
        )

        with TemporaryDirectory() as output:
            cmd = "{} fit --code-dir {} --target {} --input {} --verbose ".format(
                ArgumentsOptions.MAIN_COMMAND, custom_model_dir, self.target[problem], input_dataset
            )
            if use_output:
                cmd += " --output {}".format(output)
            if problem == BINARY:
                cmd = self._cmd_add_class_labels(cmd, framework, problem)
            if docker:
                cmd += " --docker {} ".format(docker)

            cmd += weights_cmd

            TestCMRunner._exec_shell_cmd(
                cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
            )
            TestCMRunner._delete_custom_model_dir(custom_model_dir)

    def _create_fit_input_data_dir(self, input_dir, problem, weights):
        input_dataset = self._get_dataset_filename(None, problem)
        df = pd.read_csv(input_dataset)

        # Training data
        with open(os.path.join(input_dir, "X.csv"), "w+") as fp:
            feature_df = df.loc[:, df.columns != self.target[problem]]
            feature_df.to_csv(fp, index=False)

        # Target data
        with open(os.path.join(input_dir, "y.csv"), "w+") as fp:
            target_series = df[self.target[problem]]
            target_series.to_csv(fp, index=False, header="Target")

        # Weights data
        if weights:
            df = pd.read_csv(input_dataset)
            weights_data = pd.Series(np.random.randint(1, 3, len(df)))
            with open(os.path.join(input_dir, "weights.csv"), "w+") as fp:
                weights_data.to_csv(fp, header=False)

    @pytest.mark.parametrize("framework", [SKLEARN, XGB, KERAS])
    @pytest.mark.parametrize("problem", [BINARY, REGRESSION])
    @pytest.mark.parametrize("language", [PYTHON])
    @pytest.mark.parametrize("weights", [WEIGHTS_CSV, None])
    def test_fit_sh(self, framework, problem, language, weights):
        custom_model_dir = self._create_custom_model_dir(
            framework, problem, language, is_training=True
        )
        env = os.environ
        fit_sh = os.path.join(
            self.tests_root_path,
            "..",
            "public_dropin_environments/{}_{}/fit.sh".format(language, framework),
        )
        with TemporaryDirectory() as input_dir, TemporaryDirectory() as output:
            self._create_fit_input_data_dir(input_dir, problem, weights)

            env["CODEPATH"] = custom_model_dir
            env["INPUT_DIRECTORY"] = input_dir
            env["ARTIFACT_DIRECTORY"] = output

            if problem == BINARY:
                labels = self._get_class_labels(framework, problem)
                env["NEGATIVE_CLASS_LABEL"] = labels[0]
                env["POSITIVE_CLASS_LABEL"] = labels[1]
            else:
                if os.environ.get("NEGATIVE_CLASS_LABEL"):
                    del os.environ["NEGATIVE_CLASS_LABEL"]
                    del os.environ["POSITIVE_CLASS_LABEL"]

            TestCMRunner._exec_shell_cmd(fit_sh, "Failed cmd {}".format(fit_sh), env=env)
            TestCMRunner._delete_custom_model_dir(custom_model_dir)


class TestDrumRuntime:
    @classmethod
    def setup_class(cls):
        TestCMRunner.setup_class()

    Options = collections.namedtuple(
        "Options", "force_start_internal address", defaults=["localhost"]
    )

    @mock.patch("datarobot_drum.drum.runtime.run_error_server")
    def test_no_exceptions(self, mock_run_error_server):
        with DrumRuntime():
            pass

        mock_run_error_server.assert_not_called()

    @mock.patch("datarobot_drum.drum.runtime.run_error_server")
    def test_exception_no_options(self, mock_run_error_server):
        with pytest.raises(ValueError):
            with DrumRuntime():
                raise ValueError()

        mock_run_error_server.assert_not_called()

    @mock.patch("datarobot_drum.drum.runtime.run_error_server")
    def test_exception_initialization_succeeded(self, mock_run_error_server):
        with pytest.raises(ValueError):
            with DrumRuntime() as runtime:
                runtime.options = TestDrumRuntime.Options(False)
                runtime.initialization_succeeded = True
                raise ValueError()

        mock_run_error_server.assert_not_called()

    @mock.patch("datarobot_drum.drum.runtime.run_error_server")
    def test_exception_no_force_start(self, mock_run_error_server):
        with pytest.raises(ValueError):
            with DrumRuntime() as runtime:
                runtime.options = TestDrumRuntime.Options(False)
                runtime.initialization_succeeded = False
                raise ValueError()

        mock_run_error_server.assert_not_called()

    @mock.patch("datarobot_drum.drum.runtime.run_error_server")
    def test_exception_force_start(self, mock_run_error_server):
        with pytest.raises(ValueError):
            with DrumRuntime() as runtime:
                runtime.options = TestDrumRuntime.Options(True)
                runtime.initialization_succeeded = False
                raise ValueError()

        mock_run_error_server.assert_called()

    @pytest.fixture(
        params=[
            # (REGRESSION, DOCKER_PYTHON_SKLEARN),
            (BINARY, None),
        ]
    )
    def params(self, request):
        framework = SKLEARN
        language = PYTHON

        problem, docker = request.param

        custom_model_dir = TestCMRunner._create_custom_model_dir(framework, problem, language)

        server_run_args = dict(
            framework=framework,
            problem=problem,
            language=language,
            docker=docker,
            custom_model_dir=custom_model_dir,
        )

        return framework, problem, language, docker, custom_model_dir, server_run_args

    def assert_drum_server_run_failure(self, server_run_args, force_start, error_message):
        drum_server_run = DrumServerRun(**server_run_args, force_start=force_start)

        if force_start:
            # assert that error the server is up and message is propagated via API
            with drum_server_run as run:
                response = requests.post(run.url_server_address + "/predict/")

                assert response.status_code == 503
                assert "ERROR: {}".format(error_message) in response.json()["message"]
        else:
            # DrumServerRun tries to ping the server.
            # assert that the process is already dead we it's done.
            with pytest.raises(ProcessLookupError), drum_server_run:
                pass

        assert drum_server_run.process.returncode == 1
        assert error_message in drum_server_run.process.err_stream

    @pytest.mark.parametrize("force_start", [False, True])
    def test_e2e_no_model_artifact(self, params, force_start):
        """
        Verify that if an error occurs on drum server initialization if no model artifact is found
          - if '--force-start-internal' is not set, drum server process will exit with error
          - if '--force-start-internal' is set, 'error server' will still be started, and
            will be serving initialization error
        """
        framework, problem, language, docker, custom_model_dir, server_run_args = params

        error_message = "Could not find model artifact file"

        # remove model artifact
        for item in os.listdir(custom_model_dir):
            if item.endswith(PythonArtifacts.PKL_EXTENSION):
                os.remove(os.path.join(custom_model_dir, item))

        self.assert_drum_server_run_failure(server_run_args, force_start, error_message)

    @pytest.mark.parametrize("force_start", [False, True])
    def test_e2e_model_loading_fails(self, params, force_start):
        """
        Verify that if an error occurs on drum server initialization if model cannot load properly
          - if '--force-start-internal' is not set, drum server process will exit with error
          - if '--force-start-internal' is set, 'error server' will still be started, and
            will be serving initialization error
        """
        framework, problem, language, docker, custom_model_dir, server_run_args = params

        error_message = (
            "Could not find any framework to handle loaded model and a score hook is not provided"
        )

        # make model artifact invalid by erasing its content
        for item in os.listdir(custom_model_dir):
            if item.endswith(PythonArtifacts.PKL_EXTENSION):
                with open(os.path.join(custom_model_dir, item), "wb") as f:
                    f.write(pickle.dumps("invalid model content"))

        self.assert_drum_server_run_failure(server_run_args, force_start, error_message)

    @pytest.mark.parametrize("force_start", [False, True])
    def test_e2e_predict_fails(self, params, force_start):
        """
        Verify that if an error occurs when drum server is started on /predict/ route,
        regardless '--force-start-internal' flag 'error server' will is not started.
        """
        framework, problem, language, docker, custom_model_dir, server_run_args = params

        # remove a module required during processing of /predict/ request
        os.remove(os.path.join(custom_model_dir, "custom.py"))

        drum_server_run = DrumServerRun(**server_run_args, force_start=force_start)

        with drum_server_run as run:
            input_dataset = TestCMRunner._get_dataset_filename(framework, problem)

            response = requests.post(
                run.url_server_address + "/predict/", files={"X": open(input_dataset)}
            )

            assert response.status_code == 500  # error occurs

            # assert that 'error server' is not started.
            # as 'error server' propagates errors with 503 status code,
            # assert that after error occurred, the next request is not 503
            response = requests.post(run.url_server_address + "/predict/")

            error_message = "ERROR: Samples should be provided as a csv file under `X` key."
            assert response.status_code == 422
            assert response.json()["message"] == error_message

        assert drum_server_run.process.returncode == 0
