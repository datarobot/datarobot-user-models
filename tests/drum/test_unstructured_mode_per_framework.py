"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import pytest
import requests
import werkzeug

from datarobot_drum.drum.enum import ArgumentsOptions
from datarobot_drum.drum.server import HTTP_422_UNPROCESSABLE_ENTITY

from datarobot_drum.resource.drum_server_utils import DrumServerRun

from datarobot_drum.resource.utils import (
    _exec_shell_cmd,
    _create_custom_model_dir,
)

from requests_toolbelt import MultipartEncoder

from tests.drum.constants import (
    R_NO_ARTIFACTS,
    SKLEARN_NO_ARTIFACTS,
    PYTHON_UNSTRUCTURED,
    PYTHON_UNSTRUCTURED_PARAMS,
    R_UNSTRUCTURED,
    R_UNSTRUCTURED_PARAMS,
    UNSTRUCTURED,
)

from tests.conftest import skip_if_framework_not_in_env

UTF8 = "utf8"
UTF16 = "utf16"


class TestUnstructuredMode:
    @pytest.mark.parametrize("mimetype", [None, "text/plain", "any_other_is_binary"])
    @pytest.mark.parametrize("ret_mode", ["text", "binary"])
    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN_NO_ARTIFACTS, UNSTRUCTURED, PYTHON_UNSTRUCTURED, None),
            (R_NO_ARTIFACTS, UNSTRUCTURED, R_UNSTRUCTURED, None),
        ],
    )
    def test_unstructured_models_batch(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
        mimetype,
        ret_mode,
        tmp_path,
        framework_env,
    ):
        skip_if_framework_not_in_env(framework, framework_env)
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        input_dataset = resources.datasets(framework, problem)

        output = tmp_path / "output"

        content_type = "--content-type '{};'".format(mimetype) if mimetype is not None else ""
        cmd = "{} score --code-dir {} --input {} --output {} --target-type unstructured {} --query 'ret_mode={}'".format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            output,
            content_type,
            ret_mode,
        )

        if docker:
            cmd += " --docker {} --verbose ".format(docker)

        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
        if ret_mode == "binary":
            with open(output, "rb") as f:
                out_data = f.read()
                assert 10 == int.from_bytes(out_data, byteorder="big")
        else:
            with open(output) as f:
                out_data = f.read()
                assert "10" in out_data

    @pytest.mark.parametrize(
        "framework, problem, language, nginx, docker",
        [
            (SKLEARN_NO_ARTIFACTS, UNSTRUCTURED, PYTHON_UNSTRUCTURED, False, None),
            (R_NO_ARTIFACTS, UNSTRUCTURED, R_UNSTRUCTURED, False, None),
        ],
    )
    def test_custom_models_with_drum_prediction_server(
        self,
        resources,
        framework,
        problem,
        language,
        nginx,
        docker,
        tmp_path,
        framework_env,
    ):
        skip_if_framework_not_in_env(framework, framework_env)
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        with DrumServerRun(
            "unstructured",
            resources.class_labels(framework, problem),
            custom_model_dir,
            docker,
            nginx=nginx,
        ) as run:
            input_dataset = resources.datasets(framework, problem)

            for ret_mode in ["text", "binary"]:
                for endpoint in ["/predictUnstructured/", "/predictionsUnstructured/"]:
                    # do predictions
                    url = run.url_server_address + endpoint
                    data = open(input_dataset, "rb").read()
                    params = {"ret_mode": ret_mode}
                    headers = {"Content-Type": "application/x-www-urlencoded"}
                    response = requests.post(url=url, headers=headers, data=data, params=params)

                    assert response.ok
                    if ret_mode == "text":
                        assert response.text == "10"
                    else:
                        assert 10 == int.from_bytes(response.content, byteorder="big")

    @pytest.mark.parametrize(
        "framework, problem, language",
        [
            (SKLEARN_NO_ARTIFACTS, UNSTRUCTURED, PYTHON_UNSTRUCTURED),
        ],
    )
    def test_unstructured_model_multipart_form_data_with_drum_prediction_server(
        self,
        resources,
        framework,
        problem,
        language,
        tmp_path,
        framework_env,
    ):
        skip_if_framework_not_in_env(framework, framework_env)
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        with DrumServerRun(
            "unstructured",
            resources.class_labels(framework, problem),
            custom_model_dir,
        ) as run:
            input_dataset = resources.datasets(framework, problem)

            for endpoint in ["/predictUnstructured/", "/predictionsUnstructured/"]:
                # do predictions
                mp = MultipartEncoder({"filekey": ("filename.txt", open(input_dataset, "rb"))})
                headers = {"Content-Type": mp.content_type}
                url = run.url_server_address + endpoint
                params = {"ret_mode": "text"}
                response = requests.post(url=url, headers=headers, data=mp, params=params)

                assert response.ok
                assert response.text == "10"

    @pytest.mark.parametrize(
        "framework, problem, language",
        [
            (SKLEARN_NO_ARTIFACTS, UNSTRUCTURED, PYTHON_UNSTRUCTURED),
        ],
    )
    def test_unstructured_mode_prediction_server_wrong_endpoint(
        self,
        resources,
        framework,
        problem,
        language,
        tmp_path,
        framework_env,
    ):
        skip_if_framework_not_in_env(framework, framework_env)
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        with DrumServerRun(
            "unstructured",
            resources.class_labels(framework, problem),
            custom_model_dir,
        ) as run:
            for endpoint in ["/predict/", "/predictions/"]:
                response = requests.post(url=run.url_server_address + endpoint)
                assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY
                expected_msg = "ERROR: This model has target type 'unstructured', use the /predictUnstructured/ or /predictionsUnstructured/ endpoint."
                assert response.json()["message"] == expected_msg

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN_NO_ARTIFACTS, UNSTRUCTURED, PYTHON_UNSTRUCTURED_PARAMS, None),
            (R_NO_ARTIFACTS, UNSTRUCTURED, R_UNSTRUCTURED_PARAMS, None),
        ],
    )
    def test_response_content_type(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
        tmp_path,
        framework_env,
    ):
        skip_if_framework_not_in_env(framework, framework_env)
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        with DrumServerRun(
            "unstructured",
            resources.class_labels(framework, problem),
            custom_model_dir,
            docker,
        ) as run:
            text_data = "my text, мой текст"

            # Fixtures are not used as don't want to spin up server for each test case
            # Test case with "application/octet-stream" is not very correct as data is returned as text.
            # In this test data is sent with mimetype=text/plain, so score_unstructured receives data as text.
            # Hook returns data as text with ret_charset, so response data will be encoded with this charset.
            for request_charset in [None, UTF8, UTF16]:
                for ret_charset in [None, UTF8, UTF16]:
                    for ret_mimetype in ["application/octet-stream", "text/plain_drum_test"]:
                        for endpoint in ["/predictUnstructured/", "/predictionsUnstructured/"]:
                            params = {}
                            params["ret_one_or_two"] = "two"
                            charset_to_encode = UTF8 if request_charset is None else request_charset
                            # do predictions
                            url = run.url_server_address + endpoint
                            headers = {
                                "Content-Type": "text/plain; charset={}".format(charset_to_encode)
                            }
                            if ret_charset is not None:
                                params["ret_charset"] = ret_charset
                            if ret_mimetype is not None:
                                params["ret_mimetype"] = ret_mimetype
                            response = requests.post(
                                url=url,
                                data=text_data.encode(charset_to_encode),
                                params=params,
                                headers=headers,
                            )

                            expected_charset = UTF8 if ret_charset is None else ret_charset
                            assert response.ok
                            content_type_header = response.headers["Content-Type"]
                            assert ret_mimetype in content_type_header
                            assert "charset={}".format(expected_charset) in content_type_header
                            assert text_data == response.content.decode(expected_charset)

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN_NO_ARTIFACTS, UNSTRUCTURED, PYTHON_UNSTRUCTURED_PARAMS, None),
            (R_NO_ARTIFACTS, UNSTRUCTURED, R_UNSTRUCTURED_PARAMS, None),
        ],
    )
    # In this test hook returns only data value or a tuple (data, None),
    # Check Content-Type header value.
    # Incoming data is sent back.
    def test_response_one_var_return(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
        tmp_path,
        framework_env,
    ):
        skip_if_framework_not_in_env(framework, framework_env)
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        with DrumServerRun(
            "unstructured",
            resources.class_labels(framework, problem),
            custom_model_dir,
            docker,
        ) as run:
            url = run.url_server_address + "/predictUnstructured/"

            for one_or_two in ["one", "one-with-none"]:
                input_dataset = resources.datasets(framework, problem)
                data_bytes = open(input_dataset, "rb").read()
                params = {"ret_one_or_two": one_or_two}

                # Sending None or text_data encoded with utf8, by default text files are opened using utf8
                # Content-Type is not used in the hook, but used by drum to decode
                # Expected response content type is default: "text/plain; charset=UTF-8"
                for data in [None, data_bytes]:
                    for ct in ["text/plain; charset=UTF-8", "text/some_other;"]:
                        for endpoint in ["/predictUnstructured/", "/predictionsUnstructured/"]:
                            url = run.url_server_address + endpoint
                            headers = {"Content-Type": ct}
                            response = requests.post(
                                url=url, data=data, params=params, headers=headers
                            )
                            assert response.ok
                            content_type_header = response.headers["Content-Type"]
                            mimetype, content_type_params_dict = werkzeug.http.parse_options_header(
                                content_type_header
                            )
                            assert mimetype == "text/plain"
                            assert content_type_params_dict["charset"] == UTF8
                            if data is None:
                                assert len(response.content) == 0
                            else:
                                assert response.content == data_bytes

                # Sending text_data encoded with utf16.
                # Content-Type is not used in the hook, but used by drum to decode.
                # Expected response content type is default: "text/plain; charset=UTF-8"
                data_text = "some text текст"
                data_bytes = "some text текст".encode(UTF16)
                for data in [data_bytes]:
                    for ct in ["text/plain; charset={}".format(UTF16)]:
                        for endpoint in ["/predictUnstructured/", "/predictionsUnstructured/"]:
                            url = run.url_server_address + endpoint
                            headers = {"Content-Type": ct}
                            response = requests.post(
                                url=url, data=data, params=params, headers=headers
                            )
                            assert response.ok
                            content_type_header = response.headers["Content-Type"]
                            mimetype, content_type_params_dict = werkzeug.http.parse_options_header(
                                content_type_header
                            )
                            assert mimetype == "text/plain"
                            assert content_type_params_dict["charset"] == UTF8
                            if data is None:
                                assert len(response.content) == 0
                            else:
                                assert response.content == data_text.encode(UTF8)

                # sending binary data
                for ct in ["application/octet-stream;", "application/x-www-urlencoded;"]:
                    headers = {"Content-Type": ct}
                    response = requests.post(
                        url=url, data=data_bytes, params=params, headers=headers
                    )
                    assert response.ok
                    content_type_header = response.headers["Content-Type"]
                    mimetype, content_type_params_dict = werkzeug.http.parse_options_header(
                        content_type_header
                    )
                    assert "application/octet-stream" == mimetype
                    # check params dict is empty
                    assert not any(content_type_params_dict)
                    assert response.content == data_bytes
