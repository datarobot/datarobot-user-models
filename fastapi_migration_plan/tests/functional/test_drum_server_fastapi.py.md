# Plan: tests/functional/test_drum_server_fastapi.py

Functional tests for the FastAPI production server.

## Overview

This test file mirrors `tests/functional/test_drum_server_custom_flask.py` and provides comprehensive testing for the FastAPI server implementation.

## Proposed Implementation:

```python
"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
import os
from pathlib import Path
import shutil
from typing import Generator

import pytest
import requests

from tests.constants import (
    PYTHON,
    PYTHON_UNSTRUCTURED,
    REGRESSION,
    BINARY,
    MULTICLASS,
    UNSTRUCTURED,
    TESTS_FIXTURES_PATH,
    TESTS_DATA_PATH,
)
from datarobot_drum.drum.root_predictors.drum_server_utils import DrumServerRun
from datarobot_drum.drum.root_predictors.utils import _create_custom_model_dir
from datarobot_drum.drum.utils.drum_utils import unset_drum_supported_env_vars


class TestDrumServerFastAPI:
    """Basic FastAPI server functionality tests."""
    
    @pytest.fixture(scope="class")
    def custom_model_dir(self, resources, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("model_dir")
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_dir,
            None,
            REGRESSION,
            PYTHON,
        )
        return custom_model_dir

    @pytest.fixture(scope="class")
    def drum_server(self, resources, custom_model_dir) -> Generator[DrumServerRun, None, None]:
        unset_drum_supported_env_vars()
        # Set server type to fastapi
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        try:
            with DrumServerRun(
                resources.target_types(REGRESSION),
                resources.class_labels(None, REGRESSION),
                custom_model_dir,
            ) as run:
                yield run
        finally:
            os.environ.pop("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE", None)

    def test_ping_endpoint(self, drum_server):
        """Test /ping/ endpoint returns 200 OK."""
        response = requests.get(drum_server.url_server_address + "/ping/")
        assert response.ok
        assert response.json() == {"message": "OK"}

    def test_root_endpoint(self, drum_server):
        """Test / endpoint returns 200 OK (same as ping)."""
        response = requests.get(drum_server.url_server_address + "/")
        assert response.ok

    def test_info_endpoint(self, drum_server):
        """Test /info/ endpoint returns server type fastapi."""
        response = requests.get(drum_server.url_server_address + "/info/")
        assert response.ok
        data = response.json()
        assert data["drumServer"] == "fastapi"
        assert "drumVersion" in data
        assert "language" in data

    def test_health_endpoint(self, drum_server):
        """Test /health/ endpoint returns 200 OK."""
        response = requests.get(drum_server.url_server_address + "/health/")
        assert response.ok
        assert response.json() == {"message": "OK"}

    def test_capabilities_endpoint(self, drum_server):
        """Test /capabilities/ endpoint returns capabilities dict."""
        response = requests.get(drum_server.url_server_address + "/capabilities/")
        assert response.ok
        data = response.json()
        assert isinstance(data, dict)

    def test_stats_endpoint(self, drum_server):
        """Test /stats/ endpoint returns resource info."""
        response = requests.get(drum_server.url_server_address + "/stats/")
        assert response.ok
        data = response.json()
        assert "mem_info" in data or "time_info" in data


class TestDrumServerFastAPIPredict:
    """FastAPI prediction endpoint tests."""
    
    @pytest.fixture(scope="class")
    def custom_model_dir(self, resources, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("model_dir")
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_dir,
            None,
            REGRESSION,
            PYTHON,
        )
        return custom_model_dir

    @pytest.fixture(scope="class")
    def drum_server(self, resources, custom_model_dir) -> Generator[DrumServerRun, None, None]:
        unset_drum_supported_env_vars()
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        try:
            with DrumServerRun(
                resources.target_types(REGRESSION),
                resources.class_labels(None, REGRESSION),
                custom_model_dir,
            ) as run:
                yield run
        finally:
            os.environ.pop("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE", None)

    def test_predict_endpoint_csv(self, drum_server):
        """Test /predict/ endpoint with CSV data."""
        test_file = Path(TESTS_DATA_PATH) / "juniors_3_year_stats_regression.csv"
        with open(test_file, "rb") as f:
            response = requests.post(
                drum_server.url_server_address + "/predict/",
                files={"X": f},
            )
        assert response.ok

    def test_predictions_endpoint(self, drum_server):
        """Test /predictions/ endpoint (alias for /predict/)."""
        test_file = Path(TESTS_DATA_PATH) / "juniors_3_year_stats_regression.csv"
        with open(test_file, "rb") as f:
            response = requests.post(
                drum_server.url_server_address + "/predictions/",
                files={"X": f},
            )
        assert response.ok

    def test_invocations_endpoint(self, drum_server):
        """Test /invocations endpoint (SageMaker compatible)."""
        test_file = Path(TESTS_DATA_PATH) / "juniors_3_year_stats_regression.csv"
        with open(test_file, "rb") as f:
            response = requests.post(
                drum_server.url_server_address + "/invocations",
                files={"X": f},
            )
        assert response.ok


class TestDrumServerFastAPIUnstructured:
    """FastAPI unstructured prediction tests."""
    
    @pytest.fixture(scope="class")
    def custom_model_dir(self, resources, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("model_dir")
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_dir,
            None,
            UNSTRUCTURED,
            PYTHON_UNSTRUCTURED,
        )
        return custom_model_dir

    @pytest.fixture(scope="class")
    def drum_server(self, resources, custom_model_dir) -> Generator[DrumServerRun, None, None]:
        unset_drum_supported_env_vars()
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        try:
            with DrumServerRun(
                resources.target_types(UNSTRUCTURED),
                resources.class_labels(None, UNSTRUCTURED),
                custom_model_dir,
            ) as run:
                yield run
        finally:
            os.environ.pop("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE", None)

    def test_predict_unstructured_endpoint(self, drum_server):
        """Test /predictUnstructured/ endpoint."""
        response = requests.post(
            drum_server.url_server_address + "/predictUnstructured/",
            data="test data",
            headers={"Content-Type": "text/plain"},
        )
        assert response.ok

    def test_predictions_unstructured_endpoint(self, drum_server):
        """Test /predictionsUnstructured/ endpoint (alias)."""
        response = requests.post(
            drum_server.url_server_address + "/predictionsUnstructured/",
            data="test data",
            headers={"Content-Type": "text/plain"},
        )
        assert response.ok


class TestDrumServerFastAPIVersioning:
    """Test API versioning parity."""

    @pytest.fixture(scope="class")
    def custom_model_dir(self, resources, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("model_dir_chat")
        from tests.constants import PYTHON_CHAT
        return _create_custom_model_dir(resources, tmp_dir, None, UNSTRUCTURED, PYTHON_CHAT)

    @pytest.fixture(scope="class")
    def drum_server(self, resources, custom_model_dir) -> Generator[DrumServerRun, None, None]:
        unset_drum_supported_env_vars()
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        try:
            with DrumServerRun(
                resources.target_types(UNSTRUCTURED),
                resources.class_labels(None, UNSTRUCTURED),
                custom_model_dir,
            ) as run:
                yield run
        finally:
            os.environ.pop("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE", None)

    def test_v1_chat_completions_parity(self, drum_server):
        """Test that /v1/chat/completions works identically to /chat/completions."""
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        }
        # Test original endpoint
        resp1 = requests.post(drum_server.url_server_address + "/chat/completions", json=payload)
        # Test v1 alias
        resp2 = requests.post(drum_server.url_server_address + "/v1/chat/completions", json=payload)
        
        assert resp1.status_code == resp2.status_code == 200
        assert resp1.json() == resp2.json()


class TestDrumServerFastAPICustomAuth:
    """FastAPI custom extension (auth) tests."""
    
    @pytest.fixture(scope="class")
    def custom_fastapi_script(self):
        return (Path(TESTS_FIXTURES_PATH) / "custom_fastapi_demo_auth.py", "custom_fastapi.py")

    @pytest.fixture(scope="class")
    def custom_model_dir(self, custom_fastapi_script, resources, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("model_dir")
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_dir,
            None,
            UNSTRUCTURED,
            PYTHON_UNSTRUCTURED,
        )
        fixture_filename, target_name = custom_fastapi_script
        shutil.copy2(fixture_filename, custom_model_dir / target_name)
        return custom_model_dir

    @pytest.fixture(scope="class")
    def drum_server(self, resources, custom_model_dir) -> Generator[DrumServerRun, None, None]:
        unset_drum_supported_env_vars()
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        try:
            with DrumServerRun(
                resources.target_types(UNSTRUCTURED),
                resources.class_labels(None, UNSTRUCTURED),
                custom_model_dir,
            ) as run:
                yield run
        finally:
            os.environ.pop("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE", None)

    def test_auth_passthrough(self, drum_server):
        """Test that ping endpoint bypasses auth."""
        response = requests.get(drum_server.url_server_address + "/ping/")
        assert response.ok

    def test_missing_auth_header(self, drum_server):
        """Test that missing auth header returns 401."""
        response = requests.get(drum_server.url_server_address + "/info/")
        assert response.status_code == 401
        assert response.json()["message"] == "Missing X-Auth header"

    def test_bad_auth_token(self, drum_server):
        """Test that invalid auth token returns 401."""
        response = requests.get(
            drum_server.url_server_address + "/info/", headers={"X-Auth": "token"}
        )
        assert response.status_code == 401
        assert response.json()["message"] == "Auth token is invalid"

    def test_successful_auth(self, drum_server):
        """Test that valid auth token returns 200."""
        response = requests.get(
            drum_server.url_server_address + "/info/", headers={"X-Auth": "t0k3n"}
        )
        assert response.ok
        assert response.json()["drumServer"] == "fastapi"


class TestDrumServerFastAPIErrorHandling:
    """FastAPI error handling tests."""
    
    @pytest.fixture(scope="class")
    def custom_model_dir(self, resources, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("model_dir")
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_dir,
            None,
            REGRESSION,
            PYTHON,
        )
        return custom_model_dir

    @pytest.fixture(scope="class")
    def drum_server(self, resources, custom_model_dir) -> Generator[DrumServerRun, None, None]:
        unset_drum_supported_env_vars()
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        try:
            with DrumServerRun(
                resources.target_types(REGRESSION),
                resources.class_labels(None, REGRESSION),
                custom_model_dir,
            ) as run:
                yield run
        finally:
            os.environ.pop("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE", None)

    def test_404_not_found(self, drum_server):
        """Test that unknown endpoint returns 404."""
        response = requests.get(drum_server.url_server_address + "/unknown_endpoint/")
        assert response.status_code == 404

    def test_invalid_predict_data(self, drum_server):
        """Test that invalid prediction data returns error."""
        response = requests.post(
            drum_server.url_server_address + "/predict/",
            data="invalid data",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code >= 400

    def test_request_body_too_large(self, drum_server):
        """Test that requests exceeding DRUM_FASTAPI_MAX_UPLOAD_SIZE return 413."""
        # We can set a small limit for this test via environment variable if needed,
        # but here we test the default or configured limit.
        large_data = "x" * (100 * 1024 * 1024 + 1) # Exceed default 100MB
        response = requests.post(
            drum_server.url_server_address + "/predict/",
            data=large_data,
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 413
        assert "too large" in response.json()["message"]


class TestDrumServerErrorFormatParity:
    """
    Test that error response formats are identical between Flask and FastAPI.
    This ensures backward compatibility during migration.
    """
    
    @pytest.fixture(scope="class")
    def custom_model_dir(self, resources, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("model_dir")
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_dir,
            None,
            REGRESSION,
            PYTHON,
        )
        return custom_model_dir

    @pytest.fixture(scope="class")
    def flask_server(self, resources, custom_model_dir) -> Generator[DrumServerRun, None, None]:
        """Start a Flask server for comparison."""
        unset_drum_supported_env_vars()
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "flask"
        try:
            with DrumServerRun(
                resources.target_types(REGRESSION),
                resources.class_labels(None, REGRESSION),
                custom_model_dir,
            ) as run:
                yield run
        finally:
            os.environ.pop("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE", None)

    @pytest.fixture(scope="class")
    def fastapi_server(self, resources, custom_model_dir) -> Generator[DrumServerRun, None, None]:
        """Start a FastAPI server for comparison."""
        unset_drum_supported_env_vars()
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        try:
            with DrumServerRun(
                resources.target_types(REGRESSION),
                resources.class_labels(None, REGRESSION),
                custom_model_dir,
            ) as run:
                yield run
        finally:
            os.environ.pop("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE", None)

    def test_error_format_422_missing_data(self, flask_server, fastapi_server):
        """
        Test that 422 error format is the same for both servers.
        Expected format: {"message": "ERROR: ..."}
        """
        # Flask response
        flask_response = requests.post(
            flask_server.url_server_address + "/predict/",
            data="",
            headers={"Content-Type": "text/csv"},
        )
        
        # FastAPI response
        fastapi_response = requests.post(
            fastapi_server.url_server_address + "/predict/",
            data="",
            headers={"Content-Type": "text/csv"},
        )
        
        # Both should return same status code
        assert flask_response.status_code == fastapi_response.status_code
        
        # Both should have "message" key in response
        flask_json = flask_response.json()
        fastapi_json = fastapi_response.json()
        
        assert "message" in flask_json
        assert "message" in fastapi_json
        
        # Both messages should start with "ERROR:"
        assert flask_json["message"].startswith("ERROR:")
        assert fastapi_json["message"].startswith("ERROR:")

    def test_error_format_500_server_error(self, flask_server, fastapi_server):
        """
        Test that 500 error format is the same for both servers.
        Expected format: {"message": "ERROR: ..."}
        """
        # This test requires a way to trigger a 500 error
        # We'll skip if we can't trigger one reliably
        pass  # TODO: Implement when we have a reliable way to trigger 500

    def test_error_format_wrong_target_type(self, flask_server, fastapi_server):
        """
        Test error format when calling wrong endpoint for target type.
        E.g., calling /transform/ on a regression model.
        """
        # Flask response
        flask_response = requests.post(
            flask_server.url_server_address + "/transform/",
            files={"X": ("test.csv", b"a,b\n1,2")},
        )
        
        # FastAPI response
        fastapi_response = requests.post(
            fastapi_server.url_server_address + "/transform/",
            files={"X": ("test.csv", b"a,b\n1,2")},
        )
        
        # Both should return 422
        assert flask_response.status_code == 422
        assert fastapi_response.status_code == 422
        
        flask_json = flask_response.json()
        fastapi_json = fastapi_response.json()
        
        # Both should have same format
        assert "message" in flask_json
        assert "message" in fastapi_json
        
        # Both messages should indicate wrong target type
        assert "target type" in flask_json["message"].lower()
        assert "target type" in fastapi_json["message"].lower()

    def test_success_response_format_parity(self, flask_server, fastapi_server):
        """
        Test that successful prediction response format is the same.
        """
        test_file = Path(TESTS_DATA_PATH) / "juniors_3_year_stats_regression.csv"
        
        with open(test_file, "rb") as f:
            flask_response = requests.post(
                flask_server.url_server_address + "/predict/",
                files={"X": f},
            )
        
        with open(test_file, "rb") as f:
            fastapi_response = requests.post(
                fastapi_server.url_server_address + "/predict/",
                files={"X": f},
            )
        
        assert flask_response.status_code == 200
        assert fastapi_response.status_code == 200
        
        flask_json = flask_response.json()
        fastapi_json = fastapi_response.json()
        
        # Both should have "predictions" key
        assert "predictions" in flask_json
        assert "predictions" in fastapi_json
        
        # Predictions should have same length
        assert len(flask_json["predictions"]) == len(fastapi_json["predictions"])

    def test_info_response_format_parity(self, flask_server, fastapi_server):
        """
        Test that /info/ response format is the same (except drumServer value).
        """
        flask_response = requests.get(flask_server.url_server_address + "/info/")
        fastapi_response = requests.get(fastapi_server.url_server_address + "/info/")
        
        assert flask_response.status_code == 200
        assert fastapi_response.status_code == 200
        
        flask_json = flask_response.json()
        fastapi_json = fastapi_response.json()
        
        # Both should have same keys (except drumServer has different value)
        assert set(flask_json.keys()) == set(fastapi_json.keys())
        
        # drumServer should differ
        assert flask_json["drumServer"] == "flask"
        assert fastapi_json["drumServer"] == "fastapi"
        
        # Other values should be the same
        for key in flask_json.keys():
            if key != "drumServer":
                assert flask_json[key] == fastapi_json[key], f"Mismatch for key: {key}"


@pytest.mark.parametrize("server_type", ["flask", "fastapi"])
class TestDrumServerBothTypes:
    """
    Parametrized tests that run against both Flask and FastAPI.
    This ensures feature parity between implementations.
    """
    
    @pytest.fixture(scope="class")
    def custom_model_dir(self, resources, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("model_dir")
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_dir,
            None,
            REGRESSION,
            PYTHON,
        )
        return custom_model_dir

    @pytest.fixture(scope="function")
    def drum_server(self, resources, custom_model_dir, server_type) -> Generator[DrumServerRun, None, None]:
        unset_drum_supported_env_vars()
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = server_type
        try:
            with DrumServerRun(
                resources.target_types(REGRESSION),
                resources.class_labels(None, REGRESSION),
                custom_model_dir,
            ) as run:
                yield run
        finally:
            os.environ.pop("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE", None)

    def test_ping_returns_ok(self, drum_server, server_type):
        """Test ping endpoint on both servers."""
        response = requests.get(drum_server.url_server_address + "/ping/")
        assert response.ok
        assert response.json() == {"message": "OK"}

    def test_health_returns_ok(self, drum_server, server_type):
        """Test health endpoint on both servers."""
        response = requests.get(drum_server.url_server_address + "/health/")
        assert response.ok

    def test_predict_works(self, drum_server, server_type):
        """Test prediction endpoint on both servers."""
        test_file = Path(TESTS_DATA_PATH) / "juniors_3_year_stats_regression.csv"
        with open(test_file, "rb") as f:
            response = requests.post(
                drum_server.url_server_address + "/predict/",
                files={"X": f},
            )
        assert response.ok
        assert "predictions" in response.json()

    def test_request_id_header_returned(self, drum_server, server_type):
        """Test request ID header is returned on both servers."""
        response = requests.get(drum_server.url_server_address + "/ping/")
        # Check for header (case-insensitive)
        headers_lower = {k.lower(): v for k, v in response.headers.items()}
        assert "x_request_id" in headers_lower or "x-request-id" in headers_lower

    def test_drum_version_header_returned(self, drum_server, server_type):
        """Test DRUM version header is returned on both servers."""
        response = requests.get(drum_server.url_server_address + "/ping/")
        headers_lower = {k.lower(): v for k, v in response.headers.items()}
        assert "x-drum-version" in headers_lower


class TestDrumServerFastAPIRequestHeaders:
    """Test request/response headers in FastAPI."""
    
    @pytest.fixture(scope="class")
    def custom_model_dir(self, resources, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("model_dir")
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_dir,
            None,
            REGRESSION,
            PYTHON,
        )
        return custom_model_dir

    @pytest.fixture(scope="class")
    def drum_server(self, resources, custom_model_dir) -> Generator[DrumServerRun, None, None]:
        unset_drum_supported_env_vars()
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        try:
            with DrumServerRun(
                resources.target_types(REGRESSION),
                resources.class_labels(None, REGRESSION),
                custom_model_dir,
            ) as run:
                yield run
        finally:
            os.environ.pop("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE", None)

    def test_request_id_header(self, drum_server):
        """Test that X-Request-ID header is returned."""
        response = requests.get(drum_server.url_server_address + "/ping/")
        assert "X_Request_ID" in response.headers or "x-request-id" in response.headers.keys()

    def test_custom_request_id_preserved(self, drum_server):
        """Test that custom X-Request-ID is preserved."""
        custom_id = "test-request-id-12345"
        response = requests.get(
            drum_server.url_server_address + "/ping/",
            headers={"X-Request-ID": custom_id},
        )
        # Check that the same request ID is returned
        returned_id = response.headers.get("X_Request_ID") or response.headers.get("x-request-id")
        assert returned_id == custom_id

    def test_drum_version_header(self, drum_server):
        """Test that X-Drum-Version header is returned."""
        response = requests.get(drum_server.url_server_address + "/ping/")
        assert "X-Drum-Version" in response.headers or "x-drum-version" in response.headers.keys()


class TestDrumServerFastAPISSE:
    """Test SSE streaming for chat completions."""

    @pytest.fixture(scope="class")
    def custom_model_dir(self, resources, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("model_dir")
        # Use a chat-capable dummy model
        from tests.constants import PYTHON_CHAT
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_dir,
            None,
            UNSTRUCTURED,
            PYTHON_CHAT,
        )
        return custom_model_dir

    @pytest.fixture(scope="class")
    def drum_server(self, resources, custom_model_dir) -> Generator[DrumServerRun, None, None]:
        unset_drum_supported_env_vars()
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        try:
            with DrumServerRun(
                resources.target_types(UNSTRUCTURED),
                resources.class_labels(None, UNSTRUCTURED),
                custom_model_dir,
            ) as run:
                yield run
        finally:
            os.environ.pop("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE", None)

    def test_chat_streaming(self, drum_server):
        """Test chat streaming (SSE) returns event-stream."""
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True
        }
        response = requests.post(
            drum_server.url_server_address + "/v1/chat/completions",
            json=payload,
            stream=True
        )
        assert response.ok
        assert "text/event-stream" in response.headers.get("Content-Type", "")

        # Verify SSE format
        lines = []
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                lines.append(decoded_line)
                if "[DONE]" in decoded_line:
                    break
        
        assert any(l.startswith("data: ") for l in lines)
        assert any("[DONE]" in l for l in lines)


class TestDrumServerFastAPIMultipart:
    """Test multipart form-data with key 'X'."""

    @pytest.fixture(scope="class")
    def custom_model_dir(self, resources, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("model_dir")
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_dir,
            None,
            REGRESSION,
            PYTHON,
        )
        return custom_model_dir

    @pytest.fixture(scope="class")
    def drum_server(self, resources, custom_model_dir) -> Generator[DrumServerRun, None, None]:
        unset_drum_supported_env_vars()
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        try:
            with DrumServerRun(
                resources.target_types(REGRESSION),
                resources.class_labels(None, REGRESSION),
                custom_model_dir,
            ) as run:
                yield run
        finally:
            os.environ.pop("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE", None)

    def test_predict_multipart_x(self, drum_server):
        """Test prediction with CSV file uploaded via multipart form-data key 'X'."""
        test_file = Path(TESTS_DATA_PATH) / "juniors_3_year_stats_regression.csv"
        with open(test_file, "rb") as f:
            response = requests.post(
                drum_server.url_server_address + "/predict/",
                files={"X": ("test.csv", f, "text/csv")},
            )
        assert response.ok
        assert "predictions" in response.json()


class TestDrumServerFastAPIGracefulShutdown:
    """Test graceful shutdown of FastAPI server."""

    def test_graceful_shutdown_on_sigterm(self, resources, custom_model_dir):
        """Test that server shuts down gracefully on SIGTERM."""
        import signal
        import time
        
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        try:
            with DrumServerRun(
                resources.target_types(REGRESSION),
                resources.class_labels(None, REGRESSION),
                custom_model_dir,
            ) as run:
                # Send SIGTERM to the process
                run._proc.send_signal(signal.SIGTERM)
                
                # Wait for process to exit
                exit_code = run._proc.wait(timeout=10)
                assert exit_code == 0
                
                # Verify logs for graceful shutdown messages
                # (This depends on DrumServerRun capturing stdout/stderr)
                # assert "FastAPI lifespan shutdown complete" in run.get_logs()
        finally:
            os.environ.pop("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE", None)

@pytest.mark.parametrize("problem", [BINARY, MULTICLASS, ANOMALY])
class TestDrumServerFastAPIOtherTypes:
    """Test FastAPI server with other target types."""

    @pytest.fixture(scope="class")
    def custom_model_dir(self, resources, tmp_path_factory, problem):
        tmp_dir = tmp_path_factory.mktemp(f"model_dir_{problem}")
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_dir,
            SKLEARN,
            problem,
            PYTHON,
        )
        return custom_model_dir

    @pytest.fixture(scope="class")
    def drum_server(self, resources, custom_model_dir, problem) -> Generator[DrumServerRun, None, None]:
        unset_drum_supported_env_vars()
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        try:
            with DrumServerRun(
                resources.target_types(problem),
                resources.class_labels(SKLEARN, problem),
                custom_model_dir,
            ) as run:
                yield run
        finally:
            os.environ.pop("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE", None)

    def test_predict(self, drum_server, resources, problem):
        """Test prediction for various problem types."""
        test_file = resources.get_test_dataset(problem)
        with open(test_file, "rb") as f:
            response = requests.post(
                drum_server.url_server_address + "/predict/",
                files={"X": f},
            )
        assert response.ok
        assert "predictions" in response.json()

class TestDrumServerFastAPIURLPrefix:
    """Test FastAPI server with URL prefix."""

    @pytest.fixture(scope="class")
    def custom_model_dir(self, resources, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("model_dir_prefix")
        return _create_custom_model_dir(resources, tmp_dir, None, REGRESSION, PYTHON)

    @pytest.fixture(scope="class")
    def drum_server(self, resources, custom_model_dir) -> Generator[DrumServerRun, None, None]:
        unset_drum_supported_env_vars()
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        os.environ["URL_PREFIX"] = "/test-prefix"
        try:
            with DrumServerRun(
                resources.target_types(REGRESSION),
                resources.class_labels(None, REGRESSION),
                custom_model_dir,
            ) as run:
                yield run
        finally:
            os.environ.pop("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE", None)
            os.environ.pop("URL_PREFIX", None)

    def test_prefixed_ping(self, drum_server):
        """Test ping with URL prefix."""
        response = requests.get(drum_server.url_server_address + "/test-prefix/ping/")
        assert response.ok
        assert response.json() == {"message": "OK"}

@pytest.mark.parametrize("language", [R_LANG, JULIA])
class TestDrumServerFastAPIOtherLanguages:
    """Test FastAPI server with R and Julia."""

    @pytest.fixture(scope="class")
    def custom_model_dir(self, resources, tmp_path_factory, language):
        tmp_dir = tmp_path_factory.mktemp(f"model_dir_{language}")
        return _create_custom_model_dir(resources, tmp_dir, None, REGRESSION, language)

    @pytest.fixture(scope="class")
    def drum_server(self, resources, custom_model_dir, language) -> Generator[DrumServerRun, None, None]:
        unset_drum_supported_env_vars()
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        try:
            with DrumServerRun(
                resources.target_types(REGRESSION),
                resources.class_labels(None, REGRESSION),
                custom_model_dir,
            ) as run:
                yield run
        finally:
            os.environ.pop("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE", None)

    def test_predict(self, drum_server):
        """Test prediction with other languages."""
        test_file = Path(TESTS_DATA_PATH) / "juniors_3_year_stats_regression.csv"
        with open(test_file, "rb") as f:
            response = requests.post(
                drum_server.url_server_address + "/predict/",
                files={"X": f},
            )
        assert response.ok
        assert "predictions" in response.json()
```

## Test Parity with Flask

To ensure full parity, the following existing tests should also be parameterized to run against FastAPI:

| Test File | Description |
|-----------|-------------|
| `test_drum_server_failures.py` | Server failure scenarios |
| `test_stats.py` | Statistics collection |
| `test_deployment_config.py` | Deployment configuration |
| `test_dr_api_access.py` | DataRobot API access |
| `test_mlops_monitoring.py` | MLOps monitoring integration |
| `test_runtime_parameters.py` | Runtime parameters handling |

## Running Tests

```bash
# Run FastAPI-specific tests
pytest tests/functional/test_drum_server_fastapi.py -v

# Run with specific server type
MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE=fastapi pytest tests/functional/test_drum_server_fastapi.py -v
```

## Notes:
- Tests use the `MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE` environment variable to select FastAPI.
- The fixture cleanup ensures the environment variable is removed after tests.
- Request/response header tests verify middleware functionality.
- `TestDrumServerErrorFormatParity` ensures error responses are identical between Flask and FastAPI.
- `TestDrumServerBothTypes` runs the same tests against both server types using `pytest.mark.parametrize`.
- Error format parity is critical for backward compatibility with existing clients.
