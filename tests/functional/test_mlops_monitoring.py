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
import re
from subprocess import TimeoutExpired
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

from datarobot_drum.resource.utils import (
    _exec_shell_cmd,
    _cmd_add_class_labels,
    _create_custom_model_dir,
)
from .utils import SimpleCache


class TestMLOpsMonitoring:
    @pytest.fixture
    def mask_mlops_installation(self):
        try:
            import datarobot.mlops.mlops as mlops

            mlops_filepath = os.path.abspath(mlops.__file__)
            tmp_mlops_filepath = mlops_filepath + ".tmp"
            os.rename(mlops_filepath, tmp_mlops_filepath)
            yield
            os.rename(tmp_mlops_filepath, mlops_filepath)
        except ImportError:
            yield

    @contextlib.contextmanager
    def local_webserver_stub(self, expected_pred_requests_queries=0):
        init_cache_data = {"actual_version_queries": 0, "actual_pred_requests_queries": 0}
        with SimpleCache(init_cache_data) as cache:
            app = Flask(__name__)

            @app.route("/api/v2/version/")
            def version():
                cache.inc_value("actual_version_queries")
                return json.dumps({"major": 2, "minor": 28, "versionString": "2.28.0"}), 200

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
                assert cache_data["actual_version_queries"] == 3
                if expected_pred_requests_queries:
                    assert (
                        cache_data["actual_pred_requests_queries"] == expected_pred_requests_queries
                    )

            _verify_expected_queries()

            proc.terminate()

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
            mlops_spool_dir = tmp_path / "mlops_spool"
            os.mkdir(str(mlops_spool_dir))

            # spooler_type is case-insensitive in the datarobot-mlops==8.3.0
            monitor_settings = (
                "spooler_type={};directory={};max_files=1;file_max_size=1024000".format(
                    "FILESYSTEM" if is_embedded else "filesystem", mlops_spool_dir
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

        return cmd, input_dataset, output, mlops_spool_dir

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None),
        ],
    )
    def test_drum_regression_model_monitoring_with_mlops_installed(
        self, resources, framework, problem, language, docker, tmp_path
    ):
        cmd, input_file, output_file, mlops_spool_dir = TestMLOpsMonitoring._drum_with_monitoring(
            resources, framework, problem, language, docker, tmp_path
        )

        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
        in_data = pd.read_csv(input_file)
        out_data = pd.read_csv(output_file)
        assert in_data.shape[0] == out_data.shape[0]

        print("Spool dir {}".format(mlops_spool_dir))
        assert os.path.isdir(mlops_spool_dir)
        assert os.path.isfile(os.path.join(mlops_spool_dir, "fs_spool.1"))

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
        cmd, _, _, _ = TestMLOpsMonitoring._drum_with_monitoring(
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
        cmd, _, _, _ = TestMLOpsMonitoring._drum_with_monitoring(
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
    @pytest.mark.parametrize(
        "with_monitor_settings",
        [False, True],
    )
    def test_drum_unstructured_model_embedded_monitoring(
        self, resources, framework, problem, language, tmp_path, with_monitor_settings
    ):
        cmd, _, output_file, mlops_spool_dir = TestMLOpsMonitoring._drum_with_monitoring(
            resources,
            framework,
            problem,
            language,
            docker=None,
            tmp_path=tmp_path,
            is_embedded=True,
            with_monitor_settings=with_monitor_settings,
        )

        if with_monitor_settings:
            assert os.path.exists(mlops_spool_dir)
        else:
            assert mlops_spool_dir is None

        with self.local_webserver_stub():
            _exec_shell_cmd(
                cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
            )

        with open(output_file) as f:
            out_data = f.read()
            assert "10" in out_data

    @pytest.mark.parametrize(
        "framework, problem, language",
        [(None, UNSTRUCTURED, PYTHON_UNSTRUCTURED_MLOPS)],
    )
    def test_drum_unstructured_model_embedded_monitoring_in_sklearn_env(
        self, resources, framework, problem, language, tmp_path
    ):
        cmd, _, output_file, mlops_spool_dir = TestMLOpsMonitoring._drum_with_monitoring(
            resources,
            framework,
            problem,
            language,
            docker=None,
            tmp_path=tmp_path,
            is_embedded=True,
            # Only test without explicitly provided monitor settings. Spooler folder will be created by default by DRUM.
            # If explicitly provide spooler folder, it should be mapped in `docker run`
            with_monitor_settings=False,
        )

        cmd += " --docker {}/{}".format(PUBLIC_DROPIN_ENVS_PATH, PYTHON_SKLEARN)

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
        docker_cmd_lst.insert(5, "-p")
        docker_cmd_lst.insert(6, "13909:13909")

        with self.local_webserver_stub():
            _exec_shell_cmd(docker_cmd_lst, "Failed in command line! {}".format(docker_cmd_lst))

        with open(output_file) as f:
            out_data = f.read()
            assert "10" in out_data

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

        mlops_spool_dir = tmp_path / "mlops_spool"
        os.mkdir(str(mlops_spool_dir))
        monitor_settings = (
            "spooler_type=FILESYSTEM;directory={};max_files=1;file_max_size=1024000".format(
                mlops_spool_dir
            )
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
            f' --monitor-settings "{monitor_settings}"'
        )

        with self.local_webserver_stub(expected_pred_requests_queries=10):
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
