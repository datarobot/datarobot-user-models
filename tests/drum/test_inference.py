import json
from tempfile import NamedTemporaryFile

import io
import pandas as pd
import pyarrow
import pytest
import requests
import scipy


from datarobot_drum.drum.common import (
    ArgumentsOptions,
    PredictionServerMimetypes,
    X_TRANSFORM_KEY,
    Y_TRANSFORM_KEY,
)
from datarobot_drum.resource.transform_helpers import (
    read_arrow_payload,
    read_mtx_payload,
    read_csv_payload,
    validate_transformed_output,
    parse_multi_part_response,
)
from .constants import (
    BINARY,
    CODEGEN,
    DOCKER_PYTHON_SKLEARN,
    KERAS,
    MOJO,
    MULTI_ARTIFACT,
    MULTICLASS,
    MULTICLASS_BINARY,
    NO_CUSTOM,
    POJO,
    PYPMML,
    PYTHON,
    PYTHON_LOAD_MODEL,
    PYTHON_TRANSFORM,
    PYTHON_TRANSFORM_DENSE,
    PYTHON_TRANSFORM_NO_Y,
    PYTHON_TRANSFORM_NO_Y_DENSE,
    PYTHON_XGBOOST_CLASS_LABELS_VALIDATION,
    PYTORCH,
    R,
    R_FIT,
    RDS,
    RDS_SPARSE,
    REGRESSION,
    REGRESSION_INFERENCE,
    RESPONSE_PREDICTIONS_KEY,
    SKLEARN,
    SKLEARN_TRANSFORM,
    SKLEARN_TRANSFORM_DENSE,
    SPARSE,
    TRANSFORM,
    XGB,
)
from datarobot_drum.resource.drum_server_utils import DrumServerRun
from datarobot_drum.resource.utils import (
    _cmd_add_class_labels,
    _create_custom_model_dir,
    _exec_shell_cmd,
)

from datarobot_drum.drum.utils import StructuredInputReadUtils


class TestInference:
    @pytest.fixture
    def temp_file(self):
        with NamedTemporaryFile() as f:
            yield f

    @pytest.mark.parametrize(
        "framework, problem, language, docker, use_labels_file",
        [
            (SKLEARN, REGRESSION_INFERENCE, NO_CUSTOM, None, False),
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN, False),
            (SKLEARN, BINARY, PYTHON, None, False),
            (SKLEARN, MULTICLASS, PYTHON, None, False),
            (SKLEARN, MULTICLASS_BINARY, PYTHON, None, False),
            (SKLEARN, MULTICLASS, PYTHON, None, True),
            (SKLEARN, MULTICLASS, PYTHON, DOCKER_PYTHON_SKLEARN, False),
            (SKLEARN, MULTICLASS, PYTHON, DOCKER_PYTHON_SKLEARN, True),
            (KERAS, REGRESSION, PYTHON, None, False),
            (KERAS, BINARY, PYTHON, None, False),
            (KERAS, MULTICLASS, PYTHON, None, False),
            (KERAS, MULTICLASS_BINARY, PYTHON, None, False),
            (XGB, REGRESSION, PYTHON, None, False),
            (XGB, BINARY, PYTHON, None, False),
            (XGB, BINARY, PYTHON_XGBOOST_CLASS_LABELS_VALIDATION, None, False),
            (XGB, MULTICLASS, PYTHON, None, False),
            (XGB, MULTICLASS, PYTHON_XGBOOST_CLASS_LABELS_VALIDATION, None, False),
            (XGB, MULTICLASS_BINARY, PYTHON, None, False),
            (PYTORCH, REGRESSION, PYTHON, None, False),
            (PYTORCH, BINARY, PYTHON, None, False),
            (PYTORCH, MULTICLASS, PYTHON, None, False),
            (PYTORCH, MULTICLASS_BINARY, PYTHON, None, False),
            (RDS, REGRESSION, R, None, False),
            (RDS, BINARY, R, None, False),
            (RDS, MULTICLASS, R, None, False),
            (RDS, MULTICLASS_BINARY, R, None, False),
            (CODEGEN, REGRESSION, NO_CUSTOM, None, False),
            (CODEGEN, BINARY, NO_CUSTOM, None, False),
            (CODEGEN, MULTICLASS, NO_CUSTOM, None, False),
            (CODEGEN, MULTICLASS_BINARY, NO_CUSTOM, None, False),
            (POJO, REGRESSION, NO_CUSTOM, None, False),
            (POJO, BINARY, NO_CUSTOM, None, False),
            (POJO, MULTICLASS, NO_CUSTOM, None, False),
            (POJO, MULTICLASS_BINARY, NO_CUSTOM, None, False),
            (MOJO, REGRESSION, NO_CUSTOM, None, False),
            (MOJO, BINARY, NO_CUSTOM, None, False),
            (MOJO, MULTICLASS, NO_CUSTOM, None, False),
            (MOJO, MULTICLASS_BINARY, NO_CUSTOM, None, False),
            (MULTI_ARTIFACT, REGRESSION, PYTHON_LOAD_MODEL, None, False),
            (PYPMML, REGRESSION, NO_CUSTOM, None, False),
            (PYPMML, BINARY, NO_CUSTOM, None, False),
            (PYPMML, MULTICLASS, NO_CUSTOM, None, False),
            (PYPMML, MULTICLASS_BINARY, NO_CUSTOM, None, False),
        ],
    )
    def test_custom_models_with_drum(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
        tmp_path,
        use_labels_file,
        temp_file,
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

        cmd = '{} score --code-dir {} --input "{}" --output {} --target-type {}'.format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
            input_dataset,
            output,
            resources.target_types(problem),
        )
        if resources.target_types(problem) in [BINARY, MULTICLASS]:
            cmd = _cmd_add_class_labels(
                cmd,
                resources.class_labels(framework, problem),
                target_type=resources.target_types(problem),
                multiclass_label_file=temp_file if use_labels_file else None,
            )
        if docker:
            cmd += " --docker {} --verbose ".format(docker)

        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
        in_data = pd.read_csv(input_dataset)
        out_data = pd.read_csv(output)
        assert in_data.shape[0] == out_data.shape[0]

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),
            (SKLEARN, BINARY, PYTHON, None),
            (SKLEARN, MULTICLASS, PYTHON, None),
            (SKLEARN, MULTICLASS_BINARY, PYTHON, None),
            (KERAS, REGRESSION, PYTHON, None),
            (KERAS, BINARY, PYTHON, None),
            (KERAS, MULTICLASS, PYTHON, None),
            (KERAS, MULTICLASS_BINARY, PYTHON, None),
            (XGB, REGRESSION, PYTHON, None),
            (XGB, BINARY, PYTHON, None),
            (XGB, MULTICLASS, PYTHON, None),
            (XGB, MULTICLASS_BINARY, PYTHON, None),
            (PYTORCH, REGRESSION, PYTHON, None),
            (PYTORCH, BINARY, PYTHON, None),
            (PYTORCH, MULTICLASS, PYTHON, None),
            (PYTORCH, MULTICLASS_BINARY, PYTHON, None),
            (RDS, REGRESSION, R, None),
            (RDS, BINARY, R, None),
            (RDS, MULTICLASS, R, None),
            (RDS, MULTICLASS_BINARY, R, None),
            (CODEGEN, REGRESSION, NO_CUSTOM, None),
            (CODEGEN, BINARY, NO_CUSTOM, None),
            (CODEGEN, MULTICLASS, NO_CUSTOM, None),
            (CODEGEN, MULTICLASS_BINARY, NO_CUSTOM, None),
            (MOJO, REGRESSION, NO_CUSTOM, None),
            (MOJO, BINARY, NO_CUSTOM, None),
            (MOJO, MULTICLASS, NO_CUSTOM, None),
            (MOJO, MULTICLASS_BINARY, NO_CUSTOM, None),
            (POJO, REGRESSION, NO_CUSTOM, None),
            (POJO, BINARY, NO_CUSTOM, None),
            (POJO, MULTICLASS, NO_CUSTOM, None),
            (POJO, MULTICLASS_BINARY, NO_CUSTOM, None),
            (MULTI_ARTIFACT, REGRESSION, PYTHON_LOAD_MODEL, None),
            (PYPMML, REGRESSION, NO_CUSTOM, None),
            (PYPMML, BINARY, NO_CUSTOM, None),
            (PYPMML, MULTICLASS, NO_CUSTOM, None),
            (PYPMML, MULTICLASS_BINARY, NO_CUSTOM, None),
        ],
    )
    def test_custom_models_with_drum_prediction_server(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
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
            resources.target_types(problem),
            resources.class_labels(framework, problem),
            custom_model_dir,
            docker,
        ) as run:
            input_dataset = resources.datasets(framework, problem)
            # do predictions
            for endpoint in ["/predict/", "/predictions/"]:
                for post_args in [
                    {"files": {"X": open(input_dataset)}},
                    {"data": open(input_dataset, "rb")},
                ]:
                    response = requests.post(run.url_server_address + endpoint, **post_args)

                    print(response.text)
                    assert response.ok
                    actual_num_predictions = len(
                        json.loads(response.text)[RESPONSE_PREDICTIONS_KEY]
                    )
                    in_data = pd.read_csv(input_dataset)
                    assert in_data.shape[0] == actual_num_predictions

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),
        ],
    )
    def test_custom_models_with_drum_nginx_prediction_server(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
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
            resources.target_types(problem),
            resources.class_labels(framework, problem),
            custom_model_dir,
            docker,
            nginx=True,
        ) as run:
            input_dataset = resources.datasets(framework, problem)

            # do predictions
            for endpoint in ["/predict/", "/predictions/"]:
                for post_args in [
                    {"files": {"X": open(input_dataset)}},
                    {"data": open(input_dataset, "rb")},
                ]:
                    response = requests.post(run.url_server_address + endpoint, **post_args)

                    assert response.ok
                    actual_num_predictions = len(
                        json.loads(response.text)[RESPONSE_PREDICTIONS_KEY]
                    )
                    in_data = pd.read_csv(input_dataset)
                    assert in_data.shape[0] == actual_num_predictions

    @pytest.mark.parametrize(
        "framework, problem, language, docker, use_arrow",
        [
            (SKLEARN_TRANSFORM_DENSE, TRANSFORM, PYTHON_TRANSFORM_DENSE, None, True),
            (SKLEARN_TRANSFORM, TRANSFORM, PYTHON_TRANSFORM, None, False),
            (SKLEARN_TRANSFORM_DENSE, TRANSFORM, PYTHON_TRANSFORM_DENSE, None, False),
            (SKLEARN_TRANSFORM, TRANSFORM, PYTHON_TRANSFORM_NO_Y, None, True),
            (SKLEARN_TRANSFORM_DENSE, TRANSFORM, PYTHON_TRANSFORM_NO_Y_DENSE, None, False),
        ],
    )
    @pytest.mark.parametrize("pass_target", [True, False])
    def test_custom_transform_server(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
        tmp_path,
        use_arrow,
        pass_target,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        with DrumServerRun(
            resources.target_types(problem),
            resources.class_labels(framework, problem),
            custom_model_dir,
            docker,
        ) as run:
            input_dataset = resources.datasets(framework, problem)
            in_data = pd.read_csv(input_dataset)

            files = {"X": open(input_dataset)}
            if pass_target:
                target_dataset = resources.targets(problem)
                files["y"] = open(target_dataset)

            if use_arrow:
                files["arrow_version"] = ".2"

            response = requests.post(run.url_server_address + "/transform/", files=files)
            assert response.ok

            parsed_response = parse_multi_part_response(response)

            if framework == SKLEARN_TRANSFORM_DENSE:
                if use_arrow:
                    transformed_out = read_arrow_payload(parsed_response, X_TRANSFORM_KEY)
                    if pass_target:
                        target_out = read_arrow_payload(parsed_response, Y_TRANSFORM_KEY)
                    assert parsed_response["out.format"] == "arrow"
                else:
                    transformed_out = read_csv_payload(parsed_response, X_TRANSFORM_KEY)
                    if pass_target:
                        target_out = read_csv_payload(parsed_response, Y_TRANSFORM_KEY)
                    assert parsed_response["out.format"] == "csv"
                actual_num_predictions = transformed_out.shape[0]
            else:
                transformed_out = read_mtx_payload(parsed_response, X_TRANSFORM_KEY)
                if pass_target:
                    # this shouldn't be sparse even though features are
                    if use_arrow:
                        target_out = read_arrow_payload(parsed_response, Y_TRANSFORM_KEY)
                    else:
                        target_out = read_csv_payload(parsed_response, Y_TRANSFORM_KEY)
                actual_num_predictions = transformed_out.shape[0]
                assert parsed_response["out.format"] == "sparse"
            validate_transformed_output(
                transformed_out, should_be_sparse=framework == SKLEARN_TRANSFORM
            )
            if pass_target:
                assert all(pd.read_csv(target_dataset) == target_out)
            assert in_data.shape[0] == actual_num_predictions

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN_TRANSFORM, TRANSFORM, PYTHON_TRANSFORM, DOCKER_PYTHON_SKLEARN),
        ],
    )
    def test_custom_transforms_with_drum_nginx_prediction_server(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
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
            resources.target_types(problem),
            resources.class_labels(framework, problem),
            custom_model_dir,
            docker,
            nginx=True,
        ) as run:
            input_dataset = resources.datasets(framework, problem)
            # do predictions
            response = requests.post(
                run.url_server_address + "/transform/", files={"X": open(input_dataset)}
            )

            assert response.ok

            in_data = pd.read_csv(input_dataset)

            parsed_response = parse_multi_part_response(response)

            transformed_mat = read_mtx_payload(parsed_response, X_TRANSFORM_KEY)
            actual_num_predictions = transformed_mat.shape[0]
            assert in_data.shape[0] == actual_num_predictions

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),
            (SKLEARN, BINARY, PYTHON, None),
            (SKLEARN, MULTICLASS, PYTHON, None),
        ],
    )
    def test_custom_models_drum_prediction_server_response(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
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
            resources.target_types(problem),
            resources.class_labels(framework, problem),
            custom_model_dir,
            docker,
        ) as run:
            input_dataset = resources.datasets(framework, problem)

            # do predictions
            for endpoint in ["/predict/", "/predictions/"]:
                for post_args in [
                    {"files": {"X": open(input_dataset)}},
                    {"data": open(input_dataset, "rb")},
                ]:
                    response = requests.post(run.url_server_address + endpoint, **post_args)

                    assert response.ok
                    response_json = json.loads(response.text)
                    assert isinstance(response_json, dict)
                    assert RESPONSE_PREDICTIONS_KEY in response_json
                    predictions_list = response_json[RESPONSE_PREDICTIONS_KEY]
                    assert isinstance(predictions_list, list)
                    assert len(predictions_list)
                    prediction_item = predictions_list[0]
                    if problem in [BINARY, MULTICLASS]:
                        assert isinstance(prediction_item, dict)
                        assert len(prediction_item) == len(
                            resources.class_labels(framework, problem)
                        )
                        assert all([isinstance(x, str) for x in prediction_item.keys()])
                        assert all([isinstance(x, float) for x in prediction_item.values()])
                    elif problem == REGRESSION:
                        assert isinstance(prediction_item, float)

    @pytest.mark.parametrize(
        "framework, problem, language, supported_payload_formats",
        [
            (SKLEARN, REGRESSION, PYTHON, {"csv": None, "mtx": None, "arrow": pyarrow.__version__}),
            (RDS, REGRESSION, R, {"csv": None, "mtx": None}),
            (CODEGEN, REGRESSION, NO_CUSTOM, {"csv": None}),
        ],
    )
    def test_predictors_supported_payload_formats(
        self,
        resources,
        framework,
        problem,
        language,
        supported_payload_formats,
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
            resources.target_types(problem),
            resources.class_labels(framework, problem),
            custom_model_dir,
        ) as run:
            response = requests.get(run.url_server_address + "/capabilities/")

            assert response.ok
            assert response.json() == {"supported_payload_formats": supported_payload_formats}

    @pytest.mark.parametrize(
        "framework, problem, language",
        [
            (SKLEARN, REGRESSION_INFERENCE, PYTHON),
        ],
    )
    # Don't run this test case with nginx as it still running from the prev test case.
    @pytest.mark.parametrize("nginx", [False])
    def test_predictions_python_arrow_mtx(
        self,
        resources,
        framework,
        problem,
        language,
        nginx,
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
            resources.target_types(problem),
            resources.class_labels(framework, problem),
            custom_model_dir,
            nginx=nginx,
        ) as run:
            input_dataset = resources.datasets(framework, problem)
            df = pd.read_csv(input_dataset)
            arrow_dataset_buf = pyarrow.ipc.serialize_pandas(df, preserve_index=False).to_pybytes()

            sink = io.BytesIO()
            scipy.io.mmwrite(sink, scipy.sparse.csr_matrix(df.values))
            mtx_dataset_buf = sink.getvalue()

            # do predictions
            for endpoint in ["/predict/", "/predictions/"]:
                for post_args in [
                    {"files": {"X": ("X.arrow", arrow_dataset_buf)}},
                    {"files": {"X": ("X.mtx", mtx_dataset_buf)}},
                    {
                        "data": arrow_dataset_buf,
                        "headers": {
                            "Content-Type": "{};".format(
                                PredictionServerMimetypes.APPLICATION_X_APACHE_ARROW_STREAM
                            )
                        },
                    },
                    {
                        "data": mtx_dataset_buf,
                        "headers": {
                            "Content-Type": "{};".format(PredictionServerMimetypes.TEXT_MTX)
                        },
                    },
                ]:
                    response = requests.post(run.url_server_address + endpoint, **post_args)

                    assert response.ok
                    actual_num_predictions = len(
                        json.loads(response.text)[RESPONSE_PREDICTIONS_KEY]
                    )
                    in_data = pd.read_csv(input_dataset)
                    assert in_data.shape[0] == actual_num_predictions

    @pytest.mark.parametrize(
        "framework, problem, language",
        [
            (RDS_SPARSE, REGRESSION, R_FIT),
        ],
    )
    @pytest.mark.parametrize("nginx", [False, True])
    def test_predictions_r_mtx(
        self,
        resources,
        framework,
        problem,
        language,
        nginx,
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
            resources.target_types(problem),
            resources.class_labels(framework, problem),
            custom_model_dir,
            nginx=nginx,
        ) as run:
            input_dataset = resources.datasets(framework, SPARSE)

            # do predictions
            for endpoint in ["/predict/", "/predictions/"]:
                for post_args in [
                    {"files": {"X": ("X.mtx", open(input_dataset))}},
                    {
                        "data": open(input_dataset),
                        "headers": {
                            "Content-Type": "{};".format(PredictionServerMimetypes.TEXT_MTX)
                        },
                    },
                ]:
                    response = requests.post(run.url_server_address + endpoint, **post_args)

                    assert response.ok
                    actual_num_predictions = len(
                        json.loads(response.text)[RESPONSE_PREDICTIONS_KEY]
                    )
                    in_data = StructuredInputReadUtils.read_structured_input_file_as_df(
                        input_dataset
                    )
                    assert in_data.shape[0] == actual_num_predictions
