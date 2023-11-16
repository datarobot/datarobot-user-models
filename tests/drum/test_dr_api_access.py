"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import contextlib
import json
import multiprocessing
import os
import socket

import pytest
from flask import Flask
from flask import request
from retry import retry

from datarobot_drum.drum.enum import ArgumentsOptions

from datarobot_drum.resource.utils import _create_custom_model_dir
from datarobot_drum.resource.utils import _exec_shell_cmd
from tests.constants import PYTHON_UNSTRUCTURED_DR_API_ACCESS
from tests.constants import UNSTRUCTURED
from tests.drum.utils import SimpleCache


def _extract_token_from_header():
    return request.headers.get("Authorization").replace("Token ", "")


class TestDrApiAccess:
    """Contains cases to test DataRobot API access."""

    WEBSERVER_HOST = "localhost"
    WEBSERVER_PORT = 13919
    API_TOKEN = "zzz123"

    @pytest.fixture
    def webserver_port(self):
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    @contextlib.contextmanager
    def local_webserver_stub(self, webserver_port, expected_version_queries=1):
        init_cache_data = {"actual_ping_queries": 0, "actual_version_queries": 0, "token": ""}
        with SimpleCache(init_cache_data) as cache:
            app = Flask(__name__)

            @app.route("/ping/")
            def ping():
                cache.inc_value("actual_ping_queries")
                return json.dumps({"response": "pong", "token": _extract_token_from_header()}), 200

            @app.route("/api/v2/version/")
            def version():
                saved_content = cache.read_cache()
                saved_content["token"] = _extract_token_from_header()
                saved_content["actual_version_queries"] += 1
                cache.save_cache(saved_content)
                return json.dumps({"major": 2, "minor": 28, "versionString": "2.28.0"}), 200

            proc = multiprocessing.Process(
                target=lambda: app.run(
                    host=self.WEBSERVER_HOST,
                    port=webserver_port,
                    debug=True,
                    use_reloader=False,
                )
            )
            proc.start()

            try:
                yield

                @retry((AssertionError,), delay=1, tries=1)
                def _verify_expected_queries():
                    cache_data = cache.read_cache()
                    assert cache_data["token"] == self.API_TOKEN
                    # The actual number of version queries include the one that is done by
                    # DataRobot python client.
                    assert cache_data["actual_version_queries"] == 1 + expected_version_queries

                _verify_expected_queries()
            finally:
                proc.terminate()

    @classmethod
    def _drum_with_dr_api_access(
        cls,
        webserver_port,
        resources,
        framework,
        problem,
        language,
        tmp_path,
        allow_dr_api_access=False,
        desired_num_pings=0,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        input_filepath = f"{custom_model_dir}/input.txt"
        with open(input_filepath, "w") as f:
            f.write(str(desired_num_pings))

        output = tmp_path / "output"

        cmd = "{} score --code-dir {} --input {} --output {} --target-type {}".format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_filepath,
            output,
            resources.target_types(problem),
        )
        if allow_dr_api_access:
            cmd += (
                " --allow-dr-api-access "
                f"--webserver http://{cls.WEBSERVER_HOST}:{webserver_port} "
                f"--api-token {cls.API_TOKEN}"
            )

        return cmd

    @pytest.mark.parametrize(
        "framework, problem, language",
        [(None, UNSTRUCTURED, PYTHON_UNSTRUCTURED_DR_API_ACCESS)],
    )
    def test_no_dr_api_access(
        self, webserver_port, resources, framework, problem, language, tmp_path
    ):
        desired_version_queries = 0
        cmd = self._drum_with_dr_api_access(
            webserver_port,
            resources,
            framework,
            problem,
            language,
            tmp_path,
            allow_dr_api_access=True,
            desired_num_pings=desired_version_queries,
        )

        with self.local_webserver_stub(
            webserver_port, expected_version_queries=desired_version_queries
        ):
            _exec_shell_cmd(
                cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
            )

    @pytest.mark.parametrize(
        "framework, problem, language",
        [(None, UNSTRUCTURED, PYTHON_UNSTRUCTURED_DR_API_ACCESS)],
    )
    def test_dr_api_access_success_via_cli_input_args(
        self, webserver_port, resources, framework, problem, language, tmp_path
    ):
        desired_num_version_queries = 3
        cmd = self._drum_with_dr_api_access(
            webserver_port,
            resources,
            framework,
            problem,
            language,
            tmp_path,
            allow_dr_api_access=True,
            desired_num_pings=desired_num_version_queries,
        )

        with self.local_webserver_stub(
            webserver_port, expected_version_queries=desired_num_version_queries
        ):
            _exec_shell_cmd(
                cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
            )

    @pytest.mark.parametrize(
        "framework, problem, language",
        [(None, UNSTRUCTURED, PYTHON_UNSTRUCTURED_DR_API_ACCESS)],
    )
    def test_dr_api_access_success_via_env_vars(
        self, webserver_port, resources, framework, problem, language, tmp_path
    ):
        desired_num_pings = 2
        cmd = self._drum_with_dr_api_access(
            webserver_port,
            resources,
            framework,
            problem,
            language,
            tmp_path,
            allow_dr_api_access=False,
            desired_num_pings=desired_num_pings,
        )

        with self.local_webserver_stub(webserver_port, expected_version_queries=desired_num_pings):
            try:
                os.environ.update(
                    {
                        "ALLOW_DR_API_ACCESS_FOR_ALL_CUSTOM_MODELS": "True",
                        "EXTERNAL_WEB_SERVER_URL": f"http://{self.WEBSERVER_HOST}:{webserver_port}",
                        "API_TOKEN": self.API_TOKEN,
                    }
                )

                _exec_shell_cmd(
                    cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
                )
            finally:
                os.environ.pop("ALLOW_DR_API_ACCESS_FOR_ALL_CUSTOM_MODELS", None)
                os.environ.pop("EXTERNAL_WEB_SERVER_URL", None)
                os.environ.pop("API_TOKEN", None)
