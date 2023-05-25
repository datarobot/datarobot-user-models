"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
import os
import time
from argparse import Namespace

import pytest
import responses

from datarobot_drum.drum.model_metadata import read_model_metadata_yaml
from datarobot_drum.drum.enum import MODEL_CONFIG_FILENAME, TargetType, RunMode
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.push import (
    drum_push,
    _push_training,
    _push_inference,
    setup_validation_options,
)


@pytest.fixture
def mock_version_response():
    return {
        "id": "1",
        "custom_model_id": "1",
        "version_minor": 1,
        "version_major": 1,
        "is_frozen": False,
        "items": [{"id": "1", "file_name": "hi", "file_path": "hi", "file_source": "hi"}],
    }


@pytest.fixture
def mock_tasks_version_response():
    return {
        "id": "1",
        "custom_task_id": "1",
        "version_minor": 1,
        "version_major": 1,
        "is_frozen": False,
        "items": [
            {
                "id": "1",
                "file_name": "hi",
                "file_path": "hi",
                "file_source": "hi",
                "created": str(time.time()),
            }
        ],
        "label": "test",
        "created": str(time.time()),
    }


@pytest.fixture
def mock_custom_model_version(model_id, mock_version_response):
    responses.add(
        responses.GET,
        "http://yess/version/",
        json={"major": 2, "versionString": "2.21", "minor": 21},
        status=200,
    )
    responses.add(
        responses.POST,
        "http://yess/customModels/{}/versions/".format(model_id),
        json=mock_version_response,
        status=200,
    )
    yield


@pytest.fixture
def mock_custom_task_version(model_id, mock_tasks_version_response):
    responses.add(
        responses.POST,
        "http://yess/customTasks/{}/versions/".format(model_id),
        json=mock_tasks_version_response,
        status=200,
    )


@pytest.fixture
def mock_get_model_generator(model_id, mock_version_response):
    def _mock_get_model_generator(model_type="training", target_type="Regression"):
        body = {
            "customModelType": model_type,
            "id": model_id,
            "name": "1",
            "description": "1",
            "targetType": target_type,
            "deployments_count": "1",
            "created_by": "1",
            "updated": "1",
            "created": "1",
            "latestVersion": mock_version_response,
        }
        if model_type == "inference":
            body["language"] = "Python"
            body["trainingDataAssignmentInProgress"] = False
        responses.add(
            responses.GET,
            "http://yess/customModels/{}/".format(model_id),
            json=body,
        )
        responses.add(
            responses.POST,
            "http://yess/customModels/".format(model_id),
            json=body,
        )

    return _mock_get_model_generator


@pytest.fixture
def mock_post_blueprint(model_id):
    responses.add(
        responses.POST,
        "http://yess/userBlueprints/fromCustomTaskVersionId/",
        json={
            "blender": False,
            "blueprintId": "1",
            "diagram": "{}",
            "features": ["Custom task"],
            "featuresText": "Custom task",
            "icons": [4],
            "insights": "NA",
            "modelType": "Custom task",
            "referenceModel": False,
            "supportsGpu": True,
            "shapSupport": False,
            "supportsNewSeries": False,
            "userBlueprintId": "2",
            "userId": "userid02302312312",
            "blueprintContext": {"warnings": [], "errors": []},
            "vertexContext": [
                {
                    "information": {
                        "inputs": [
                            "Missing Values: Forbidden",
                            "Data Type: Numeric",
                            "Sparsity: Supported",
                        ],
                        "outputs": [
                            "Missing Values: Never",
                            "Data Type: Numeric",
                            "Sparsity: Never",
                        ],
                    },
                    "messages": {
                        "warnings": ["Unexpected input type. Expected Numeric, received All."]
                    },
                    "taskId": "1",
                }
            ],
            "supportedTargetTypes": ["binary"],
            "isTimeSeries": False,
            "hexColumnNameLookup": [],
            "customTaskVersionMetadata": [],
            "decompressedFormat": False,
        },
    )
    responses.add(
        responses.POST,
        "http://yess/customTasks/",
        json={
            "id": model_id,
            "target_type": "Regression",
            "created": "1",
            "updated": "1",
            "name": "1",
            "description": "1",
            "language": "Python",
            "created_by": "1",
        },
    )


@pytest.fixture
def mock_post_add_to_repository():
    responses.add(
        responses.POST,
        "http://yess/userBlueprintsProjectBlueprints/",
        json={
            "addedToMenu": [{"userBlueprintId": "2", "blueprintId": "1"}],
            "notAddedToMenu": [],
            "message": "All blueprints successfully added to project repository.",
        },
    )


@pytest.fixture
def mock_get_env(environment_id):
    responses.add(
        responses.GET,
        "http://yess/executionEnvironments/{}/".format(environment_id),
        json={
            "id": "1",
            "name": "hi",
            "latestVersion": {"id": "hii", "environment_id": environment_id, "build_status": "yes"},
        },
    )


@pytest.fixture
def mock_train_model(project_id):
    responses.add(
        responses.POST,
        "http://yess/projects/{}/models/".format(project_id),
        json={},
        adding_headers={"Location": "the/moon"},
    )
    responses.add(
        responses.GET,
        "http://yess/projects/{}/modelJobs/the/".format(project_id),
        json={
            "is_blocked": False,
            "id": "55",
            "processes": [],
            "model_type": "fake",
            "project_id": project_id,
            "blueprint_id": "1",
        },
    )


@responses.activate
@pytest.mark.parametrize(
    "config_yaml",
    ["inference_binary_metadata_yaml_no_target_name"],
)
def test_push_no_target_name_in_yaml(request, model_id, config_yaml, tmp_path):
    config_yaml = request.getfixturevalue(config_yaml)
    config_yaml = config_yaml + "\nmodelID: {}".format(model_id)

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        f.write(config_yaml)
    config = read_model_metadata_yaml(tmp_path)

    options = Namespace(code_dir=tmp_path, model_config=config)
    with pytest.raises(DrumCommonException, match="Missing keys: \['targetName'\]"):
        drum_push(options)


@responses.activate
@pytest.mark.usefixtures(
    "mock_custom_model_version",
    "mock_custom_task_version",
    "mock_post_blueprint",
    "mock_post_add_to_repository",
    "mock_get_env",
    "mock_train_model",
)
@pytest.mark.parametrize(
    "config_yaml",
    [
        "training_metadata_yaml",
        "training_metadata_yaml_with_proj",
        "inference_metadata_yaml",
        "inference_multiclass_metadata_yaml",
        "inference_multiclass_metadata_yaml_label_file",
    ],
)
@pytest.mark.parametrize("existing_model_id", [True, None])
def test_push(
    request,
    config_yaml,
    existing_model_id,
    model_id,
    mock_get_model_generator,
    multiclass_labels,
    tmp_path,
):
    if existing_model_id:
        existing_model_id = model_id
    config_yaml = request.getfixturevalue(config_yaml)
    if existing_model_id:
        config_yaml = config_yaml + "\nmodelID: {}".format(existing_model_id)

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        f.write(config_yaml)
    config = read_model_metadata_yaml(tmp_path)

    mock_get_model_generator(
        model_type=config["type"], target_type=config["targetType"].capitalize()
    )
    push_fn = _push_training if config["type"] == "training" else _push_inference
    push_fn(config, code_dir="", endpoint="http://Yess", token="okay")

    calls = responses.calls
    custom_tasks_or_models_path = "customTasks" if push_fn == _push_training else "customModels"
    if existing_model_id is None:
        assert (
            calls[1].request.path_url == "/{}/".format(custom_tasks_or_models_path)
            and calls[1].request.method == "POST"
        )
        if config["targetType"] == TargetType.MULTICLASS.value:
            sent_labels = json.loads(calls[1].request.body)["classLabels"]
            assert sent_labels == multiclass_labels
        call_shift = 1
    else:
        call_shift = 0
    assert (
        calls[call_shift + 1].request.path_url
        == "/{}/{}/versions/".format(custom_tasks_or_models_path, model_id)
        and calls[call_shift + 1].request.method == "POST"
    )
    if push_fn == _push_training:
        assert (
            calls[call_shift + 2].request.path_url == "/userBlueprints/fromCustomTaskVersionId/"
            and calls[call_shift + 2].request.method == "POST"
        )
        if "trainingModel" in config:
            assert (
                calls[call_shift + 3].request.path_url == "/userBlueprintsProjectBlueprints/"
                and calls[call_shift + 3].request.method == "POST"
            )
            assert (
                calls[call_shift + 4].request.path_url == "/projects/abc123/models/"
                and calls[call_shift + 4].request.method == "POST"
            )
            assert len(calls) == 6 + call_shift
        else:
            assert len(calls) == 3 + call_shift
    else:
        assert len(calls) == 2 + call_shift


class TestSetupValidationOptions:
    @pytest.fixture
    def training_options(self):
        return Namespace(
            user_secrets_mount_path=None,
            user_secrets_prefix=None,
            code_dir="/a/b/c",
            model_config={
                "environmentID": "5e8c889607389fe0f466c72d",
                "name": "joe",
                "targetType": "regression",
                "type": "training",
                "validation": {"input": "hello", "targetName": "target"},
            },
        )

    @pytest.fixture
    def inference_options(self, training_options):
        training_options.model_config["type"] = "inference"
        return training_options

    def test_fit_validation_options_no_user_secrets(self, training_options):
        new_options, run_mode, command = setup_validation_options(training_options)
        assert run_mode == RunMode.FIT
        assert new_options.user_secrets_mount_path is None
        assert new_options.user_secrets_prefix is None

        assert command == [
            "drum",
            "RunMode.FIT",
            "--input",
            "/a/b/c/hello",
            "--target",
            "target",
            "--code-dir",
            "/a/b/c",
        ]

    def test_fit_validation_options_with_user_secrets(self, training_options):
        mount_path = "/x/y/z"
        training_options.user_secrets_mount_path = mount_path
        prefix = "prefix"
        training_options.user_secrets_prefix = prefix
        new_options, run_mode, command = setup_validation_options(training_options)
        assert run_mode == RunMode.FIT
        assert new_options.user_secrets_mount_path == mount_path
        assert new_options.user_secrets_prefix == prefix

        assert command == [
            "drum",
            "RunMode.FIT",
            "--input",
            "/a/b/c/hello",
            "--target",
            "target",
            "--code-dir",
            "/a/b/c",
            "--user-secrets-mount-path",
            mount_path,
            "--user-secrets-prefix",
            prefix,
        ]

    def test_inference_validation_options_no_user_secrets(self, inference_options):
        new_options, run_mode, command = setup_validation_options(inference_options)
        assert run_mode == RunMode.SCORE
        assert new_options.user_secrets_mount_path is None
        assert new_options.user_secrets_prefix is None

        assert command == ["drum", "RunMode.SCORE", "--input", "/a/b/c/hello", "-cd", "/a/b/c"]

    def test_inference_validation_options_with_user_secrets(self, inference_options):
        mount_path = "/x/y/z"
        inference_options.user_secrets_mount_path = mount_path
        prefix = "prefix"
        inference_options.user_secrets_prefix = prefix
        new_options, run_mode, command = setup_validation_options(inference_options)
        assert run_mode == RunMode.SCORE
        assert new_options.user_secrets_mount_path == mount_path
        assert new_options.user_secrets_prefix == prefix

        assert command == [
            "drum",
            "RunMode.SCORE",
            "--input",
            "/a/b/c/hello",
            "-cd",
            "/a/b/c",
            "--user-secrets-mount-path",
            mount_path,
            "--user-secrets-prefix",
            prefix,
        ]
