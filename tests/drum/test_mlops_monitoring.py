"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
import multiprocessing
import os
import re

import pytest
import pandas as pd
from flask import Flask, request

from datarobot_drum.drum.enum import ArgumentsOptions

from .constants import (
    SKLEARN,
    REGRESSION_INFERENCE,
    NO_CUSTOM,
    BINARY,
    UNSTRUCTURED,
    PYTHON_UNSTRUCTURED_MLOPS,
)

from datarobot_drum.resource.utils import (
    _exec_shell_cmd,
    _cmd_add_class_labels,
    _create_custom_model_dir,
)


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

    @pytest.fixture
    def local_webserver_stub(self):
        app = Flask(__name__)

        @app.route("/api/v2/version/")
        def version():
            return json.dumps({"major": 2, "minor": 28, "versionString": "2.28.0"}), 200

        @app.route(
            "/api/v2/deployments/<deployment_id>/predictionResults/fromJSON/", methods=["POST"]
        )
        def post_prediction_results(deployment_id):
            assert deployment_id is not None
            assert request.json is not None
            return json.dumps({"message": "ok"}), 202

        proc = multiprocessing.Process(
            target=lambda: app.run(host="localhost", port=13909, debug=True, use_reloader=False)
        )
        proc.start()
        yield
        proc.terminate()

    @staticmethod
    def _drum_with_monitoring(
        resources, framework, problem, language, docker, tmp_path, is_embedded=False
    ):
        """
        We expect the run of drum to be ok, since mlops is assumed to be installed.
        """
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
        )

        mlops_spool_dir = tmp_path / "mlops_spool"
        os.mkdir(str(mlops_spool_dir))

        input_dataset = resources.datasets(framework, problem)
        output = tmp_path / "output"

        cmd = "{} score --code-dir {} --input {} --output {} --target-type {}".format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            output,
            resources.target_types(problem),
        )
        if is_embedded:
            cmd += (
                " --monitor-embedded --model-id 555 --deployment-id 777 "
                " --webserver http://localhost:13909 --api-token aaabbb"
            )
        else:
            monitor_settings = "spooler_type=filesystem;directory={};max_files=1;file_max_size=1024000".format(
                mlops_spool_dir
            )
            cmd += ' --monitor --model-id 555 --deployment-id 777 --monitor-settings="{}"'.format(
                monitor_settings
            )

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
        "framework, problem, language, docker", [(SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None),],
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
        "framework, problem, language, docker", [(SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None),],
    )
    @pytest.mark.usefixtures("mask_mlops_installation")
    @pytest.mark.sequential
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
        "framework, problem, language, docker", [(SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None),],
    )
    def test_drum_regression_model_monitoring_fails_in_unstructured_mode(
        self, resources, framework, problem, language, docker, tmp_path
    ):
        cmd, input_file, output_file, mlops_spool_dir = TestMLOpsMonitoring._drum_with_monitoring(
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
        "framework, problem, language, docker",
        [(None, UNSTRUCTURED, PYTHON_UNSTRUCTURED_MLOPS, None)],
    )
    @pytest.mark.usefixtures("local_webserver_stub")
    def test_drum_unstructured_model_monitoring_with_mlops_installed(
        self, resources, framework, problem, language, docker, tmp_path
    ):
        cmd, input_file, output_file, mlops_spool_dir = TestMLOpsMonitoring._drum_with_monitoring(
            resources, framework, problem, language, docker, tmp_path, is_embedded=True
        )

        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )

        with open(output_file) as f:
            out_data = f.read()
            assert "10" in out_data
