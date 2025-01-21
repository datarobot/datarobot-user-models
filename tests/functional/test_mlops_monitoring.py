"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import contextlib
import json
import multiprocessing
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
import re
from subprocess import TimeoutExpired
import sys
import time


from retry import retry
import pytest
import pandas as pd
import requests
from flask import Flask, request

from datarobot_drum.drum.args_parser import CMRunnerArgsRegistry
from datarobot_drum.drum.drum import CMRunner
from datarobot_drum.drum.enum import ArgumentsOptions
from datarobot_drum.drum.enum import RunMode
from datarobot_drum.drum.runtime import DrumRuntime
from datarobot_drum.drum.language_predictors.base_language_predictor import MLOps
from tests.fixtures.unstructured_custom_mlops import NUM_REPORTED_DEPLOYMENT_STATS

from tests.constants import (
    SKLEARN,
    REGRESSION_INFERENCE,
    NO_CUSTOM,
    BINARY,
    UNSTRUCTURED,
    PYTHON_UNSTRUCTURED_MLOPS,
    PUBLIC_DROPIN_ENVS_PATH,
    PYTHON_SKLEARN,
)

from datarobot_drum.drum.root_predictors.utils import (
    _exec_shell_cmd,
    _cmd_add_class_labels,
    _create_custom_model_dir,
)
from .utils import SimpleCache


class TestMLOpsMonitoring:
    @pytest.fixture
    def mask_mlops_installation(self):
        if MLOps is None:
            yield

        mlops_path = Path(sys.modules[MLOps.__module__].__file__).parent.absolute()
        tmp_mlops_filepath = mlops_path.rename(str(mlops_path) + ".tmp")
        yield
        tmp_mlops_filepath.rename(mlops_path)

    @contextlib.contextmanager
    def local_webserver_stub(self):
        init_cache_data = {"actual_pred_requests_queries": 0}
        with SimpleCache(init_cache_data) as cache:
            app = Flask(__name__)

            @app.route(
                "/api/v2/deployments/<deployment_id>/predictionRequests/fromJSON/", methods=["POST"]
            )
            def post_prediction_requests(deployment_id):
                stats = request.get_json()
                cache.inc_value("actual_pred_requests_queries", value=len(stats["data"]))
                return json.dumps({"message": "ok"}), 202

            proc = multiprocessing.Process(
                target=lambda: app.run(host="localhost", port=13909, debug=True, use_reloader=False)
            )
            proc.start()

            yield

            @retry((AssertionError,), delay=1, tries=10)
            def _verify_expected_queries():
                cache_data = cache.read_cache()
                assert cache_data["actual_pred_requests_queries"] == NUM_REPORTED_DEPLOYMENT_STATS

            try:
                _verify_expected_queries()
            finally:
                proc.terminate()
                time.sleep(0.1)  # wait for the server to stop
                proc.kill()

    @staticmethod
    def _drum_with_monitoring(
        resources,
        framework,
        problem,
        language,
        docker,
        tmp_path,
        is_embedded=False,
        with_monitor_settings=True,
    ):
        """
        We expect the run of drum to be ok, since mlops is assumed to be installed.
        """
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        input_dataset = resources.datasets(framework, problem)
        output = tmp_path / "output"

        cmd = "{} score --code-dir {} --input {} --output {} --target-type {}".format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            output,
            resources.target_types(problem),
        )
        mlops_spool_dir = None

        cmd += " --model-id 555 --deployment-id 777"
        cmd += " --monitor-embedded" if is_embedded else " --monitor"

        if with_monitor_settings:
            if is_embedded:
                monitor_settings = "spooler_type=STDOUT"
            else:
                mlops_spool_dir = tmp_path / "mlops_spool"
                os.mkdir(str(mlops_spool_dir))

                # spooler_type is case-insensitive in the datarobot-mlops==8.3.0jjj
                monitor_settings = (
                    "spooler_type=filesystem;directory={};max_files=1;file_max_size=1024000".format(
                        mlops_spool_dir
                    )
                )
            cmd += ' --monitor-settings="{}"'.format(monitor_settings)

        if is_embedded:
            cmd += " --webserver http://localhost:13909 --api-token aaabbb"

        if problem == BINARY:
            cmd = _cmd_add_class_labels(
                cmd,
                resources.class_labels(framework, problem),
                target_type=resources.target_types(problem),
            )
        if docker:
            cmd += " --docker {} --verbose ".format(docker)

        return cmd, input_dataset, output

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None),
        ],
    )
    def test_drum_regression_model_monitoring_with_mlops_installed(
        self, resources, framework, problem, language, docker, tmp_path
    ):
        cmd, input_file, output_file = TestMLOpsMonitoring._drum_with_monitoring(
            resources, framework, problem, language, docker, tmp_path, with_monitor_settings=True
        )
        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
        in_data = pd.read_csv(input_file)
        out_data = pd.read_csv(output_file)
        assert in_data.shape[0] == out_data.shape[0]

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None),
        ],
    )
    @pytest.mark.usefixtures("mask_mlops_installation")
    def test_drum_regression_model_monitoring_no_mlops_installed(
        self, resources, framework, problem, language, docker, tmp_path
    ):
        """
        We expect the run of drum to fail since the mlops package is assumed to not be installed
        Returns
        -------

        """
        cmd, _, _ = TestMLOpsMonitoring._drum_with_monitoring(
            resources, framework, problem, language, docker, tmp_path
        )
        p, _, _ = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )
        assert (
            p.returncode != 0
        ), "drum should fail when datarobot-mlops is not installed and monitoring is requested"

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None),
        ],
    )
    def test_drum_regression_model_monitoring_fails_in_unstructured_mode(
        self, resources, framework, problem, language, docker, tmp_path
    ):
        cmd, _, _ = TestMLOpsMonitoring._drum_with_monitoring(
            resources, framework, problem, language, docker, tmp_path
        )

        cmd = re.sub(r"--target-type .*? ", "", cmd)
        cmd += " --target-type unstructured"
        _, stdo, _ = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )

        assert str(stdo).find("MLOps monitoring can not be used in unstructured mode") != -1

    @pytest.mark.parametrize(
        "framework, problem, language",
        [(None, UNSTRUCTURED, PYTHON_UNSTRUCTURED_MLOPS)],
    )
    def test_drum_unstructured_model_embedded_mlops_reporting(
        self, resources, framework, problem, language, tmp_path
    ):
        cmd, _, output_file = TestMLOpsMonitoring._drum_with_monitoring(
            resources,
            framework,
            problem,
            language,
            docker=None,
            tmp_path=tmp_path,
            is_embedded=True,
            with_monitor_settings=False,
        )

        with self.local_webserver_stub():
            _exec_shell_cmd(
                cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
            )

    @pytest.mark.parametrize(
        "framework, problem, language",
        [(None, UNSTRUCTURED, PYTHON_UNSTRUCTURED_MLOPS)],
    )
    def test_drum_unstructured_model_with_stdout_spooler_type(
        self,
        resources,
        framework,
        problem,
        language,
        tmp_path,
        capsys,
    ):
        cmd, _, output_file = TestMLOpsMonitoring._drum_with_monitoring(
            resources,
            framework,
            problem,
            language,
            docker=None,
            tmp_path=tmp_path,
            is_embedded=True,
            with_monitor_settings=True,
        )

        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
        captured = capsys.readouterr()
        messages = captured.out.split("\n")
        assert len([m for m in messages if "payload" in m]) == NUM_REPORTED_DEPLOYMENT_STATS

    @pytest.mark.parametrize(
        "framework, problem, language",
        [(None, UNSTRUCTURED, PYTHON_UNSTRUCTURED_MLOPS)],
    )
    def test_drum_unstructured_model_embedded_monitoring_in_sklearn_env(
        self, resources, framework, problem, language, tmp_path
    ):
        cmd, _, output_file = TestMLOpsMonitoring._drum_with_monitoring(
            resources,
            framework,
            problem,
            language,
            docker=None,
            tmp_path=tmp_path,
            is_embedded=True,
            with_monitor_settings=False,
        )

        py_sklearn_env_path = Path(PUBLIC_DROPIN_ENVS_PATH) / PYTHON_SKLEARN
        with self._drop_in_environment_with_drum_from_source_code(
            py_sklearn_env_path
        ) as new_py_sklearn_env_path:
            cmd += f" --docker {new_py_sklearn_env_path}"

            arg_parser = CMRunnerArgsRegistry.get_arg_parser()
            args = cmd.split()

            # drop first `drum` token from the args list to be correctly parsed by arg_parser
            options = arg_parser.parse_args(args[1:])
            CMRunnerArgsRegistry.verify_options(options)
            runtime = DrumRuntime()
            runtime.options = options
            cm_runner = CMRunner(runtime)

            # This command tries to build the image and returns cmd to start DRUM in container
            docker_cmd_lst = cm_runner._prepare_docker_command(options, RunMode.SCORE, args)

            # Configure network for the container and map the stub server port.
            # I'm not sure, I want to add the following logic into DRUM itself. It should be considered expert usage
            docker_cmd_lst.insert(3, "--net")
            docker_cmd_lst.insert(4, "host")
            # docker_cmd_lst.insert(5, "-p")
            # docker_cmd_lst.insert(6, "13909:13909")

            with self.local_webserver_stub():
                _exec_shell_cmd(docker_cmd_lst, "Failed in command line! {}".format(docker_cmd_lst))

        with open(output_file) as f:
            out_data = f.read()
            assert "10" in out_data

    @contextlib.contextmanager
    def _drop_in_environment_with_drum_from_source_code(self, drop_in_env_path: Path):
        """
        This context manager creates a temporary environment with the DRUM from the source code.
        """
        current_drum_path = Path(__file__).parent.parent.parent / "custom_model_runner"
        try:
            subprocess.run(
                ["make"],
                cwd=current_drum_path,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            print(f"Failed to build the drum: {exc.stderr.decode()}")
            raise

        files = [f.name for f in (current_drum_path / "dist").iterdir() if f.is_file()]
        if len(files) != 1:
            raise ValueError("Expected only one file in the drum dist directory")
        drum_wheel_filename = files[0]

        with tempfile.TemporaryDirectory() as temp_env_dir:
            shutil.copytree(drop_in_env_path, temp_env_dir, dirs_exist_ok=True)

            shutil.copy(current_drum_path / "dist" / drum_wheel_filename, temp_env_dir)

            dockerfile_path = Path(temp_env_dir) / "Dockerfile"

            with open(dockerfile_path, "r") as file:
                dockerfile_lines = file.readlines()

            marker_substring = "# MARK: FUNCTIONAL-TEST-ADD-HERE."
            line_index_to_insert = next(
                (
                    index
                    for index, line in enumerate(dockerfile_lines)
                    if line.startswith(marker_substring)
                ),
                None,
            )
            if line_index_to_insert is None:
                raise ValueError(f"The Dockerfile does not contain the '{marker_substring}' line")

            dockerfile_lines[line_index_to_insert + 1 : line_index_to_insert + 1] = [
                f"COPY {drum_wheel_filename} {drum_wheel_filename}\n",
                "RUN pip3 uninstall -y datarobot-drum datarobot-mlops && \\\n"
                f"    pip3 install --force-reinstall {drum_wheel_filename} && \\\n"
                f"    rm -rf {drum_wheel_filename}\n",
            ]

            with open(dockerfile_path, "w") as file:
                file.writelines(dockerfile_lines)

            yield temp_env_dir

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [(None, UNSTRUCTURED, PYTHON_UNSTRUCTURED_MLOPS, None)],
    )
    def test_graceful_shutdown_in_server_run_mode_non_production(
        self, resources, framework, problem, language, docker, tmp_path
    ):
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        pred_server_host = "localhost:13908"
        cmd = (
            f"{ArgumentsOptions.MAIN_COMMAND} server "
            f" --address {pred_server_host}"
            f" --code-dir {custom_model_dir}"
            f" --target-type {resources.target_types(problem)}"
            " --monitor-embedded"
            " --model-id 555"
            " --deployment-id 777"
            " --webserver http://localhost:13909"
            " --api-token aaabbb"
        )

        with self.local_webserver_stub():
            proc, _, _ = _exec_shell_cmd(
                cmd, err_msg=f"Failed in: {cmd}", assert_if_fail=False, capture_output=False
            )

            @retry((requests.exceptions.ConnectionError,), delay=1, tries=10)
            def _check_health():
                ping_url = f"http://{pred_server_host}"
                print(f"Trying to ping to {ping_url}")
                response = requests.get(ping_url)
                print(f"Response: {response}")
                assert response.ok and response.json()["message"] == "OK"

            try:
                start_time = time.time()
                _check_health()
                print(f"Check health took {time.time()-start_time} sec")

                message = "There are four words"
                print(f"Sending prediction ...")
                response = requests.post(
                    f"http://{pred_server_host}/predictUnstructured/",
                    headers={"Accept": "text/plain", "Content-Type": "text/plain"},
                    data=message,
                )
                assert response.ok and int(response.text) == (message.count(" ") + 1)
                print(f"Prediction succeeded")
            finally:
                proc.terminate()
                try:
                    proc.wait(7)
                except TimeoutExpired:
                    proc.kill()
