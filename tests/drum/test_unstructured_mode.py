import pytest
import requests

from datarobot_drum.drum.common import ArgumentsOptions

from tests.drum.drum_server_utils import DrumServerRun

from tests.drum.utils import (
    _exec_shell_cmd,
    _create_custom_model_dir,
)

from tests.drum.constants import (
    PYTHON_UNSTRUCTURED,
    PYTHON_UNSTRUCTURED_PARAMS,
    R_UNSTRUCTURED,
    R_UNSTRUCTURED_PARAMS,
    UNSTRUCTURED,
)


class TestUnstructuredMode:
    @pytest.mark.parametrize("mimetype", [None, "text/plain", "any_other_is_binary"])
    @pytest.mark.parametrize("ret_mode", ["text", "binary"])
    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (None, UNSTRUCTURED, PYTHON_UNSTRUCTURED, None),
            (None, UNSTRUCTURED, R_UNSTRUCTURED, None),
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
    ):
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

    @pytest.mark.parametrize("ret_mode", ["text", "binary"])
    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (None, UNSTRUCTURED, PYTHON_UNSTRUCTURED, None),
            (None, UNSTRUCTURED, R_UNSTRUCTURED, None),
        ],
    )
    def test_custom_models_with_drum_prediction_server(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
        ret_mode,
        tmp_path,
    ):
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
            input_dataset = resources.datasets(framework, problem)

            # do predictions
            url = run.url_server_address + "/predictUnstructured/"
            data = open(input_dataset, "rb").read()
            params = {"ret_mode": ret_mode}
            response = requests.post(url=url, data=data, params=params)

            assert response.ok
            if ret_mode == "text":
                assert response.text == "10"
            else:
                assert 10 == int.from_bytes(response.content, byteorder="big")

    @pytest.mark.parametrize("ret_charset", [None, "utf16"])
    # testcase with "application/octet-stream" is not correct as data is returned as text
    @pytest.mark.parametrize("ret_mimetype", ["application/octet-stream", "text/plain_drum_test"])
    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (None, UNSTRUCTURED, PYTHON_UNSTRUCTURED_PARAMS, None),
            (None, UNSTRUCTURED, R_UNSTRUCTURED_PARAMS, None),
        ],
    )
    def test_response_content_type(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
        ret_charset,
        ret_mimetype,
        tmp_path,
    ):
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

            text_data = u"my text, мой текст"
            params = {}
            params["ret_one_or_two"] = "two"

            # do predictions
            url = run.url_server_address + "/predictUnstructured/"
            headers = {"Content-Type": "text/plain; charset=UTF-8"}
            if ret_charset is not None:
                params["ret_charset"] = ret_charset
            if ret_mimetype is not None:
                params["ret_mimetype"] = ret_mimetype
            response = requests.post(
                url=url, data=text_data.encode("utf8"), params=params, headers=headers
            )

            assert response.ok
            content_type_header = response.headers["Content-Type"]
            assert ret_mimetype in content_type_header
            assert (
                "charset=utf8"
                if ret_charset is None
                else "charset={}".format(ret_charset) in content_type_header
            )

            charset_to_decode = "utf8" if ret_charset is None else ret_charset
            assert text_data == response.content.decode(charset_to_decode)

    @pytest.mark.parametrize("one_or_two", ["one", "one-with-none"])
    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (None, UNSTRUCTURED, PYTHON_UNSTRUCTURED_PARAMS, None),
            (None, UNSTRUCTURED, R_UNSTRUCTURED_PARAMS, None),
        ],
    )
    def test_response_one_var_return(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
        one_or_two,
        tmp_path,
    ):
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
            input_dataset = resources.datasets(framework, problem)

            params = {"ret_one_or_two": one_or_two}
            url = run.url_server_address + "/predictUnstructured/"

            # sending None data
            data = None
            headers = {"Content-Type": "text/plain; charset=UTF-8"}
            response = requests.post(url=url, data=data, params=params, headers=headers)
            content_type_header = response.headers["Content-Type"]
            assert response.ok
            assert "text/plain" in content_type_header
            assert "charset=utf8" in content_type_header
            assert len(response.content) == 0

            # sending text data
            data = open(input_dataset, "rb").read()
            response = requests.post(url=url, data=data, params=params, headers=headers)
            assert response.ok
            content_type_header = response.headers["Content-Type"]
            assert "text/plain" in content_type_header
            assert "charset=utf8" in content_type_header
            assert response.content == data

            # sending binary data
            headers = {"Content-Type": "application/octet-stream;"}
            response = requests.post(url=url, data=data, params=params, headers=headers)
            content_type_header = response.headers["Content-Type"]
            assert "application/octet-stream" in content_type_header
            assert "charset=utf8" in content_type_header
            assert response.content == data
