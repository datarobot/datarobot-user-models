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
from tempfile import NamedTemporaryFile
from threading import Thread
from unittest import mock
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import requests

from datarobot_drum.drum.args_parser import CMRunnerArgsRegistry
from datarobot_drum.drum.common import (
    ArgumentsOptions,
    CUSTOM_FILE_NAME,
    CustomHooks,
    PythonArtifacts,
    RunMode,
)
from datarobot_drum.drum.runtime import DrumRuntime

TRAINING = "training"
INFERENCE = "inference"

# Framweork keywords
XGB = "xgboost"
KERAS = "keras"
KERAS_JOBLIB = "keras_joblib"
SKLEARN = "sklearn"
SIMPLE = "simple"
PYTORCH = "pytorch"
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
R_FIT = "R_fit"
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
        self,
        framework,
        problem,
        custom_model_dir,
        docker=None,
        with_error_server=False,
        show_stacktrace=True,
    ):
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
        if with_error_server:
            cmd += " --with-error-server"
        if show_stacktrace:
            cmd += " --show-stacktrace"
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
        cls.training_templates_path = os.path.join(
            cls.tests_root_path, "..", "model_templates", "training"
        )

        cls.paths_to_training_models = {
            (PYTHON, SKLEARN): os.path.join(cls.training_templates_path, "python3_sklearn"),
            (PYTHON, SIMPLE): os.path.join(cls.training_templates_path, "simple"),
            (PYTHON, KERAS): os.path.join(cls.training_templates_path, "python3_keras_joblib"),
            (PYTHON, XGB): os.path.join(cls.training_templates_path, "python3_xgboost"),
            (R_FIT, RDS): os.path.join(cls.training_templates_path, "r_lang"),
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
            R_FIT: (os.path.join(cls.tests_fixtures_path, "fit_custom.R"), "custom.R"),
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
            (XGB, REGRESSION): os.path.join(cls.tests_artifacts_path, "xgb_reg.pkl"),
            (XGB, BINARY): os.path.join(cls.tests_artifacts_path, "xgb_bin.pkl"),
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
    def _create_custom_model_dir(
        cls, custom_model_dir, framework, problem, language, is_training=False, nested=False
    ):
        if nested:
            custom_model_dir = custom_model_dir.joinpath("nested_dir")
        custom_model_dir.mkdir(parents=True, exist_ok=True)
        if is_training:
            model_template_dir = cls.paths_to_training_models[(language, framework)]

            if language == PYTHON:
                files = glob.glob(r"{}/*.py".format(model_template_dir))
            elif language in [R, R_ALL_HOOKS, R_FIT]:
                files = glob.glob(r"{}/*.r".format(model_template_dir)) + glob.glob(
                    r"{}/*.R".format(model_template_dir)
                )

            for filename in files:
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

    @classmethod
    def _get_artifact_filename(cls, framework, problem):
        return cls.artifacts[(framework, problem)]

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
            (XGB, REGRESSION, PYTHON, None),
            (XGB, BINARY, PYTHON, None),
            (XGB, BINARY, PYTHON_XGBOOST_CLASS_LABELS_VALIDATION, None),
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
    def test_custom_models_with_drum(self, framework, problem, language, docker, tmp_path):
        custom_model_dir = tmp_path / "custom_model"
        self._create_custom_model_dir(custom_model_dir, framework, problem, language)

        input_dataset = self._get_dataset_filename(framework, problem)

        output = tmp_path / "output"

        cmd = "{} score --code-dir {} --input {} --output {}".format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset, output
        )
        cmd = self._cmd_add_class_labels(cmd, framework, problem)
        if docker:
            cmd += " --docker {} --verbose ".format(docker)

        TestCMRunner._exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
        in_data = pd.read_csv(input_dataset)
        out_data = pd.read_csv(output)
        assert in_data.shape[0] == out_data.shape[0]

    @pytest.mark.parametrize(
        "framework, problem, language", [(SKLEARN, BINARY, PYTHON), (RDS, BINARY, R)]
    )
    def test_bin_models_with_wrong_labels(self, framework, problem, language, tmp_path):
        custom_model_dir = tmp_path / "custom_model"
        self._create_custom_model_dir(custom_model_dir, framework, problem, language)

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

        stdo_stde = str(stdo) + str(stde)

        if framework == SKLEARN:
            assert (
                str(stdo_stde).find(
                    "Wrong class labels. Use class labels detected by sklearn model"
                )
                != -1
            )
        elif framework == RDS:
            assert (
                str(stdo_stde).find(
                    "Wrong class labels. Use class labels according to your dataset"
                )
                != -1
            )

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
    def test_detect_language(self, framework, problem, language, tmp_path):
        custom_model_dir = tmp_path / "custom_model"
        self._create_custom_model_dir(custom_model_dir, framework, problem, language)

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

        stdo_stde = str(stdo) + str(stde)

        cases_1_2_3 = (
            str(stdo_stde).find("Can not detect language by artifacts and/or custom.py/R files")
            != -1
        )
        case_4 = (
            str(stdo_stde).find(
                "Could not find a serialized model artifact with .rds extension, supported by default R predictor. "
                "If your artifact is not supported by default predictor, implement custom.load_model hook."
            )
            != -1
        )
        case_5 = (
            str(stdo_stde).find(
                "Could not find model artifact file in: {} supported by default predictors".format(
                    custom_model_dir
                )
            )
            != -1
        )
        assert any([cases_1_2_3, case_4, case_5])

    @pytest.mark.parametrize(
        "framework, language", [(SKLEARN, PYTHON_ALL_HOOKS), (RDS, R_ALL_HOOKS)]
    )
    def test_custom_model_with_all_predict_hooks(self, framework, language, tmp_path):
        custom_model_dir = tmp_path / "custom_model"
        self._create_custom_model_dir(custom_model_dir, framework, REGRESSION, language)

        input_dataset = self._get_dataset_filename(framework, REGRESSION)

        output = tmp_path / "output"

        cmd = "{} score --code-dir {} --input {} --output {}".format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset, output
        )
        TestCMRunner._exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
        preds = pd.read_csv(output)
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
            (XGB, REGRESSION, PYTHON, None),
            (XGB, BINARY, PYTHON, None),
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
    def test_custom_models_with_drum_prediction_server(
        self, framework, problem, language, docker, tmp_path
    ):
        custom_model_dir = tmp_path / "custom_model"
        TestCMRunner._create_custom_model_dir(custom_model_dir, framework, problem, language)

        with DrumServerRun(framework, problem, custom_model_dir, docker) as run:
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
        self, framework, problem, language, docker, tmp_path
    ):
        custom_model_dir = tmp_path / "custom_model"
        TestCMRunner._create_custom_model_dir(custom_model_dir, framework, problem, language)

        with DrumServerRun(framework, problem, custom_model_dir, docker) as run:
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
    def test_custom_models_perf_test(self, framework, problem, language, docker, tmp_path):
        custom_model_dir = tmp_path / "custom_model"
        self._create_custom_model_dir(custom_model_dir, framework, problem, language)

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

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, BINARY, PYTHON, None),
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None),
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, DOCKER_PYTHON_SKLEARN),
        ],
    )
    def test_custom_models_validation_test(self, framework, problem, language, docker, tmp_path):
        custom_model_dir = tmp_path / "custom_model"
        self._create_custom_model_dir(custom_model_dir, framework, problem, language)

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

        if language == NO_CUSTOM:
            assert re.search(r"Null value imputation\s+FAILED", stdo)
        else:
            assert re.search(r"Null value imputation\s+PASSED", stdo)

    @pytest.mark.parametrize("language, language_suffix", [("python", ".py"), ("r", ".R")])
    def test_template_creation(self, language, language_suffix, tmp_path):
        print("Running template creation tests: {}".format(language))
        directory = tmp_path / "template_test_{}".format(uuid4())

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

    @pytest.mark.parametrize("framework", [RDS, SKLEARN, XGB, KERAS])
    @pytest.mark.parametrize("problem", [BINARY, REGRESSION])
    @pytest.mark.parametrize("docker", [DOCKER_PYTHON_SKLEARN, None])
    @pytest.mark.parametrize("weights", [WEIGHTS_CSV, WEIGHTS_ARGS, None])
    @pytest.mark.parametrize("use_output", [True, False])
    @pytest.mark.parametrize("nested", [True, False])
    def test_fit(self, framework, problem, docker, weights, use_output, tmp_path, nested):

        if framework == RDS:
            language = R_FIT
        else:
            language = PYTHON

        custom_model_dir = tmp_path / "custom_model"
        self._create_custom_model_dir(
            custom_model_dir,
            framework,
            problem,
            language,
            is_training=True,
            nested=nested if language == PYTHON else False,  # TODO: support nested R files
        )

        input_dataset = self._get_dataset_filename(framework, problem)

        weights_cmd, input_dataset, __keep_this_around = self._add_weights_cmd(
            weights, input_dataset
        )

        output = tmp_path / "output"
        output.mkdir()

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

    def _create_fit_input_data_dir(self, input_dir, problem, weights):
        input_dir.mkdir(parents=True, exist_ok=True)

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
    def test_fit_sh(self, framework, problem, language, weights, tmp_path):
        custom_model_dir = tmp_path / "custom_model"
        self._create_custom_model_dir(
            custom_model_dir, framework, problem, language, is_training=True
        )

        env = os.environ
        fit_sh = os.path.join(
            self.tests_root_path,
            "..",
            "public_dropin_environments/{}_{}/fit.sh".format(language, framework),
        )

        input_dir = tmp_path / "input_dir"
        self._create_fit_input_data_dir(input_dir, problem, weights)

        output = tmp_path / "output"
        output.mkdir()

        env["CODEPATH"] = str(custom_model_dir)
        env["INPUT_DIRECTORY"] = str(input_dir)
        env["ARTIFACT_DIRECTORY"] = str(output)

        if problem == BINARY:
            labels = self._get_class_labels(framework, problem)
            env["NEGATIVE_CLASS_LABEL"] = labels[0]
            env["POSITIVE_CLASS_LABEL"] = labels[1]
        else:
            if os.environ.get("NEGATIVE_CLASS_LABEL"):
                del os.environ["NEGATIVE_CLASS_LABEL"]
                del os.environ["POSITIVE_CLASS_LABEL"]

        TestCMRunner._exec_shell_cmd(fit_sh, "Failed cmd {}".format(fit_sh), env=env)

    def test_fit_simple(self, tmp_path):
        custom_model_dir = tmp_path / "custom_model"
        self._create_custom_model_dir(
            custom_model_dir, SIMPLE, REGRESSION, PYTHON, is_training=True, nested=True
        )

        input_dataset = self._get_dataset_filename(SKLEARN, REGRESSION)

        output = tmp_path / "output"
        output.mkdir()

        cmd = "{} fit --code-dir {} --target {} --input {} --verbose ".format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, self.target[REGRESSION], input_dataset
        )
        TestCMRunner._exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )


class TestDrumRuntime:
    @classmethod
    def setup_class(cls):
        TestCMRunner.setup_class()

    Options = collections.namedtuple(
        "Options",
        "with_error_server {} docker address verbose show_stacktrace".format(
            CMRunnerArgsRegistry.SUBPARSER_DEST_KEYWORD
        ),
        defaults=[RunMode.SERVER, None, "localhost", False, True],
    )

    class StubDrumException(Exception):
        pass

    @mock.patch("datarobot_drum.drum.runtime.run_error_server")
    def test_no_exceptions(self, mock_run_error_server):
        with DrumRuntime():
            pass

        mock_run_error_server.assert_not_called()

    @mock.patch("datarobot_drum.drum.runtime.run_error_server")
    def test_exception_no_options(self, mock_run_error_server):
        with pytest.raises(TestDrumRuntime.StubDrumException):
            with DrumRuntime():
                raise TestDrumRuntime.StubDrumException()

        mock_run_error_server.assert_not_called()

    @mock.patch("datarobot_drum.drum.runtime.run_error_server")
    def test_exception_initialization_succeeded(self, mock_run_error_server):
        with pytest.raises(TestDrumRuntime.StubDrumException):
            with DrumRuntime() as runtime:
                runtime.options = TestDrumRuntime.Options(False)
                runtime.initialization_succeeded = True
                raise TestDrumRuntime.StubDrumException()

        mock_run_error_server.assert_not_called()

    @mock.patch("datarobot_drum.drum.runtime.run_error_server")
    def test_exception_not_server_mode(self, mock_run_error_server):
        with pytest.raises(TestDrumRuntime.StubDrumException):
            with DrumRuntime() as runtime:
                runtime.options = TestDrumRuntime.Options(False, RunMode.SCORE)
                runtime.initialization_succeeded = False
                raise TestDrumRuntime.StubDrumException()

        mock_run_error_server.assert_not_called()

    @mock.patch("datarobot_drum.drum.runtime.run_error_server")
    def test_exception_not_server_mode(self, mock_run_error_server):
        with pytest.raises(TestDrumRuntime.StubDrumException):
            with DrumRuntime() as runtime:
                runtime.options = TestDrumRuntime.Options(False, RunMode.SERVER, "path_to_image")
                runtime.initialization_succeeded = False
                raise TestDrumRuntime.StubDrumException()

        mock_run_error_server.assert_not_called()

    @mock.patch("datarobot_drum.drum.runtime.run_error_server")
    def test_exception_no_with_error_server(self, mock_run_error_server):
        with pytest.raises(TestDrumRuntime.StubDrumException):
            with DrumRuntime() as runtime:
                runtime.options = TestDrumRuntime.Options(False)
                runtime.initialization_succeeded = False
                raise TestDrumRuntime.StubDrumException()

        mock_run_error_server.assert_not_called()

    @mock.patch("datarobot_drum.drum.runtime.run_error_server")
    def test_exception_with_error_server(self, mock_run_error_server):
        with pytest.raises(TestDrumRuntime.StubDrumException):
            with DrumRuntime() as runtime:
                runtime.options = TestDrumRuntime.Options(True)
                runtime.initialization_succeeded = False
                raise TestDrumRuntime.StubDrumException()

        mock_run_error_server.assert_called()

    @pytest.fixture(params=[REGRESSION, BINARY])
    def params(self, request, tmp_path):
        framework = SKLEARN
        language = PYTHON

        problem = request.param

        custom_model_dir = tmp_path / "custom_model"
        TestCMRunner._create_custom_model_dir(custom_model_dir, framework, problem, language)

        server_run_args = dict(
            framework=framework, problem=problem, custom_model_dir=custom_model_dir,
        )

        return framework, problem, custom_model_dir, server_run_args

    def assert_drum_server_run_failure(self, server_run_args, with_error_server, error_message):
        drum_server_run = DrumServerRun(**server_run_args, with_error_server=with_error_server)

        if with_error_server:
            # assert that error the server is up and message is propagated via API
            with drum_server_run as run:
                # check /health/ route
                response = requests.get(run.url_server_address + "/health/")
                assert response.status_code == 513
                assert error_message in response.json()["message"]

                # check /predict/ route
                response = requests.post(run.url_server_address + "/predict/")

                assert response.status_code == 513
                assert error_message in response.json()["message"]
        else:
            # DrumServerRun tries to ping the server.
            # assert that the process is already dead we it's done.
            with pytest.raises(ProcessLookupError), drum_server_run:
                pass

        assert drum_server_run.process.returncode == 1
        assert error_message in drum_server_run.process.err_stream

    @pytest.mark.parametrize("with_error_server", [False, True])
    def test_e2e_no_model_artifact(self, params, with_error_server):
        """
        Verify that if an error occurs on drum server initialization if no model artifact is found
          - if '--with-error-server' is not set, drum server process will exit with error
          - if '--with-error-server' is set, 'error server' will still be started, and
            will be serving initialization error
        """
        _, _, custom_model_dir, server_run_args = params

        error_message = "Could not find model artifact file"

        # remove model artifact
        for item in os.listdir(custom_model_dir):
            if item.endswith(PythonArtifacts.PKL_EXTENSION):
                os.remove(os.path.join(custom_model_dir, item))

        self.assert_drum_server_run_failure(server_run_args, with_error_server, error_message)

    @pytest.mark.parametrize("with_error_server", [False, True])
    def test_e2e_model_loading_fails(self, params, with_error_server):
        """
        Verify that if an error occurs on drum server initialization if model cannot load properly
          - if '--with-error-server' is not set, drum server process will exit with error
          - if '--with-error-server' is set, 'error server' will still be started, and
            will be serving initialization error
        """
        _, _, custom_model_dir, server_run_args = params

        error_message = (
            "Could not find any framework to handle loaded model and a score hook is not provided"
        )

        # make model artifact invalid by erasing its content
        for item in os.listdir(custom_model_dir):
            if item.endswith(PythonArtifacts.PKL_EXTENSION):
                with open(os.path.join(custom_model_dir, item), "wb") as f:
                    f.write(pickle.dumps("invalid model content"))

        self.assert_drum_server_run_failure(server_run_args, with_error_server, error_message)

    @pytest.mark.parametrize("with_error_server", [False, True])
    def test_e2e_predict_fails(self, params, with_error_server):
        """
        Verify that when drum server is started, if an error occurs on /predict/ route,
        'error server' is not started regardless '--with-error-server' flag.
        """
        framework, problem, custom_model_dir, server_run_args = params

        # remove a module required during processing of /predict/ request
        os.remove(os.path.join(custom_model_dir, "custom.py"))

        drum_server_run = DrumServerRun(**server_run_args, with_error_server=with_error_server)

        with drum_server_run as run:
            input_dataset = TestCMRunner._get_dataset_filename(framework, problem)

            response = requests.post(
                run.url_server_address + "/predict/", files={"X": open(input_dataset)}
            )

            assert response.status_code == 500  # error occurs

            # assert that 'error server' is not started.
            # as 'error server' propagates errors with 513 status code,
            # assert that after error occurred, the next request is not 513

            # check /health/ route
            response = requests.get(run.url_server_address + "/health/")
            assert response.status_code == 200

            # check /predict/ route
            response = requests.post(run.url_server_address + "/predict/")

            error_message = "ERROR: Samples should be provided as a csv file under `X` key."
            assert response.status_code == 422
            assert response.json()["message"] == error_message

        assert drum_server_run.process.returncode == 0
