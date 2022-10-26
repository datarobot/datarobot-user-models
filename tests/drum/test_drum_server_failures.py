"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import collections
import os
import pickle
import pytest
import requests

from datarobot_drum.drum.enum import PythonArtifacts, RunMode
from datarobot_drum.drum.args_parser import CMRunnerArgsRegistry

from .constants import SKLEARN, REGRESSION, BINARY, DOCKER_PYTHON_SKLEARN, PYTHON
from datarobot_drum.resource.utils import _create_custom_model_dir
from datarobot_drum.resource.drum_server_utils import DrumServerRun


class TestDrumServerFailures:
    Options = collections.namedtuple(
        "Options",
        "with_error_server {} docker address verbose show_stacktrace".format(
            CMRunnerArgsRegistry.SUBPARSER_DEST_KEYWORD
        ),
        defaults=[RunMode.SERVER, None, "localhost", False, True],
    )

    @pytest.fixture(params=[REGRESSION, BINARY])
    def params(self, resources, request, tmp_path):
        framework = SKLEARN
        language = PYTHON

        problem = request.param

        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
        )

        server_run_args = dict(
            target_type=resources.target_types(problem),
            custom_model_dir=custom_model_dir,
            labels=resources.class_labels(framework, problem),
        )

        return framework, problem, custom_model_dir, server_run_args

    def assert_drum_server_run_failure(
        self, server_run_args, with_error_server, error_message, with_nginx=False, docker=None
    ):
        drum_server_run = DrumServerRun(
            **server_run_args,
            with_error_server=with_error_server,
            nginx=with_nginx,
            docker=docker,
            fail_on_shutdown_error=False  # for now server may not exit cleanly when failed
        )

        if with_error_server or with_nginx:
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
            # if ping fails for timeout, AssertionError("Server failed to start") is risen
            with pytest.raises(AssertionError, match="Server failed to start"), drum_server_run:
                pass

            # If server is started with error server or with_nginx (in docker), it is killed in the end of test.
            # So where is no expected error message in err_stream
            assert error_message in drum_server_run.process.err_stream

    @pytest.mark.parametrize(
        "with_error_server, with_nginx, docker",
        [(False, False, None), (True, False, None), (False, True, DOCKER_PYTHON_SKLEARN)],
    )
    def test_ping_endpoints(self, params, with_error_server, with_nginx, docker):
        _, _, custom_model_dir, server_run_args = params

        # remove a module required during processing of /predict/ request
        os.remove(os.path.join(custom_model_dir, "custom.py"))

        drum_server_run = DrumServerRun(
            **server_run_args,
            with_error_server=with_error_server,
            nginx=with_nginx,
            docker=docker,
            fail_on_shutdown_error=False  # for now server may not exit cleanly when failed
        )

        with drum_server_run as run:
            response = requests.get(run.url_server_address + "/")
            assert response.status_code == 200
            response = requests.get(run.url_server_address + "/ping/")
            assert response.status_code == 200

    @pytest.mark.parametrize(
        "with_error_server, with_nginx, docker",
        [(False, False, None), (True, False, None), (False, True, DOCKER_PYTHON_SKLEARN)],
    )
    def test_e2e_no_model_artifact(self, params, with_error_server, with_nginx, docker):
        """
        Verify that if an error occurs on DRUM server initialization if no model artifact is found
          - if '--with-error-server' is not set, DRUM server process will exit with error
          - if '--with-error-server' is set, 'error server' will still be started, and
            will be serving initialization error
        """
        _, _, custom_model_dir, server_run_args = params

        error_message = "An error occurred when loading your artifact"

        # remove model artifact
        for item in os.listdir(custom_model_dir):
            if item.endswith(PythonArtifacts.PKL_EXTENSION):
                os.remove(os.path.join(custom_model_dir, item))

        self.assert_drum_server_run_failure(
            server_run_args, with_error_server, error_message, with_nginx=with_nginx, docker=docker,
        )

    @pytest.mark.parametrize(
        "with_error_server, with_nginx, docker",
        [(False, False, None), (True, False, None), (False, True, DOCKER_PYTHON_SKLEARN)],
    )
    def test_e2e_model_loading_fails(self, params, with_error_server, with_nginx, docker):
        """
        Verify that if an error occurs on DRUM server initialization if model cannot load properly
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

        self.assert_drum_server_run_failure(
            server_run_args, with_error_server, error_message, with_nginx=with_nginx, docker=docker
        )

    @pytest.mark.parametrize(
        "with_error_server, with_nginx, docker",
        [(False, False, None), (True, False, None), (False, True, DOCKER_PYTHON_SKLEARN)],
    )
    def test_e2e_predict_fails(self, resources, params, with_error_server, with_nginx, docker):
        """
        Verify that when drum server is started, if an error occurs on /predict/ route,
        'error server' is not started regardless '--with-error-server' flag.
        """
        framework, problem, custom_model_dir, server_run_args = params

        # remove a module required during processing of /predict/ request
        os.remove(os.path.join(custom_model_dir, "custom.py"))

        drum_server_run = DrumServerRun(
            **server_run_args,
            with_error_server=with_error_server,
            nginx=with_nginx,
            docker=docker,
            fail_on_shutdown_error=False  # for now server may not exit cleanly when failed
        )

        with drum_server_run as run:
            input_dataset = resources.datasets(framework, problem)

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

            error_message = (
                "ERROR: Samples should be provided as: "
                "  - a csv, mtx, or arrow file under `X` form-data param key."
                "  - binary data"
            )
            assert response.status_code == 422
            assert response.json()["message"] == error_message
