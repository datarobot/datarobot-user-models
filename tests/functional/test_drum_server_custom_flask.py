"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

from pathlib import Path
import shutil

import pytest
import requests

from tests.constants import (
    PYTHON_UNSTRUCTURED,
    UNSTRUCTURED,
    TESTS_FIXTURES_PATH,
)
from datarobot_drum.resource.drum_server_utils import DrumServerRun
from datarobot_drum.resource.utils import _create_custom_model_dir
from datarobot_drum.drum.utils.drum_utils import unset_drum_supported_env_vars


class TestDrumServerCustomAuth:
    @pytest.fixture(scope="class")
    def custom_flask_script(self):
        return (Path(TESTS_FIXTURES_PATH) / "custom_flask_demo_auth.py", "custom_flask.py")

    @pytest.fixture(scope="class")
    def custom_model_dir(self, custom_flask_script, resources, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("model_dir")
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_dir,
            None,
            UNSTRUCTURED,
            PYTHON_UNSTRUCTURED,
        )
        fixture_filename, target_name = custom_flask_script
        shutil.copy2(fixture_filename, custom_model_dir / target_name)
        return custom_model_dir

    @pytest.fixture(scope="class")
    def drum_server(self, resources, custom_model_dir):
        unset_drum_supported_env_vars()
        with DrumServerRun(
            resources.target_types(UNSTRUCTURED),
            resources.class_labels(None, UNSTRUCTURED),
            custom_model_dir,
        ) as run:
            yield run

    def test_auth_passthrough(self, drum_server):
        response = requests.get(drum_server.url_server_address + "/ping/")
        assert response.ok

    def test_missing_auth_header(self, drum_server):
        response = requests.get(drum_server.url_server_address + "/info/")
        assert response.status_code == 401
        assert response.json()["message"] == "Missing X-Auth header"

    def test_bad_auth_token(self, drum_server):
        response = requests.get(
            drum_server.url_server_address + "/info/", headers={"X-Auth": "token"}
        )
        assert response.status_code == 401
        assert response.json()["message"] == "Auth token is invalid"

    def test_successful_auth(self, drum_server):
        response = requests.get(
            drum_server.url_server_address + "/info/", headers={"X-Auth": "t0k3n"}
        )
        assert response.ok
        assert response.json()["drumServer"] == "flask"
