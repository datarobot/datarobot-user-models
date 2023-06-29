"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
import os
from tempfile import NamedTemporaryFile
import pandas as pd
import pytest
import requests

from datarobot_drum.drum.enum import ArgumentOptionsEnvVars, TargetType
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.resource.deployment_config_helpers import (
    parse_validate_deployment_config_file,
    get_class_names_from_class_mapping,
    build_pps_response_json_str,
)
from tests.drum.constants import TESTS_DEPLOYMENT_CONFIG_PATH

from tests.drum.constants import (
    BINARY,
    MULTICLASS,
    PYTHON,
    REGRESSION,
    SKLEARN,
    TEXT_GENERATION,
)

from datarobot_drum.resource.utils import _create_custom_model_dir

from datarobot_drum.resource.drum_server_utils import DrumServerRun
from datarobot_drum.drum.utils.drum_utils import unset_drum_supported_env_vars


class TestDeploymentConfig:
    deployment_config_anomaly = os.path.join(TESTS_DEPLOYMENT_CONFIG_PATH, "anomaly.json")
    deployment_config_regression = os.path.join(TESTS_DEPLOYMENT_CONFIG_PATH, "regression.json")
    deployment_config_binary = os.path.join(TESTS_DEPLOYMENT_CONFIG_PATH, "binary.json")
    deployment_config_multiclass = os.path.join(TESTS_DEPLOYMENT_CONFIG_PATH, "multiclass.json")
    deployment_config_text_generation = os.path.join(
        TESTS_DEPLOYMENT_CONFIG_PATH, "text_generation.json"
    )

    def test_parse_deployment_config_file(self):
        not_json = '{"target: {"class_mapping": null, "missing_maps_to": null, "name": "Grade 2014", "prediction_threshold": 0.5, "type": "Regression" }}'
        no_target_json = '{"targe": {"class_mapping": null, "missing_maps_to": null, "name": "Grade 2014", "prediction_threshold": 0.5, "type": "Regression" }}'

        assert parse_validate_deployment_config_file(None) is None

        with NamedTemporaryFile(mode="w+") as f:
            f.write(not_json)
            f.flush()
            with pytest.raises(
                DrumCommonException, match="Failed to parse deployment config json file"
            ):
                parse_validate_deployment_config_file(f.name)

        with NamedTemporaryFile(mode="w+") as f:
            f.write(no_target_json)
            f.flush()
            with pytest.raises(
                DrumCommonException,
                match="'target' section not found in deployment config json file",
            ):
                parse_validate_deployment_config_file(f.name)

        all_configs = [
            self.deployment_config_anomaly,
            self.deployment_config_regression,
            self.deployment_config_binary,
            self.deployment_config_multiclass,
        ]
        for config_path in all_configs:
            parse_validate_deployment_config_file(config_path)

    def test_build_pps_response_json_str_bad_target(self):
        d = {"Predictions": [1.2, 2.3, 3.4]}
        df = pd.DataFrame(data=d)
        config = parse_validate_deployment_config_file(self.deployment_config_regression)
        with pytest.raises(DrumCommonException, match="target type 'None' is not supported"):
            build_pps_response_json_str(df, config, None)

    def test_get_class_names_from_class_mappings(self):
        config = parse_validate_deployment_config_file(self.deployment_config_multiclass)
        class_names = get_class_names_from_class_mapping(config["target"]["class_mapping"])
        assert class_names == ["GALAXY", "QSO", "STAR"]

        config = parse_validate_deployment_config_file(self.deployment_config_binary)
        class_names = get_class_names_from_class_mapping(config["target"]["class_mapping"])
        assert class_names == ["Iris-versicolor", "Iris-setosa"]

    def test_map_regression_prediction(self):
        d = {"Predictions": [1.2, 2.3, 3.4]}
        df = pd.DataFrame(data=d)
        config = parse_validate_deployment_config_file(self.deployment_config_regression)
        assert config["target"]["name"] == "Grade 2014"
        assert config["target"]["type"] == "Regression"

        response = build_pps_response_json_str(df, config, TargetType.REGRESSION)
        response_json = json.loads(response)
        assert isinstance(response_json, dict)
        assert "data" in response_json
        predictions_list = response_json["data"]
        assert isinstance(predictions_list, list)
        assert len(predictions_list) == df.shape[0]

        pred_iter = iter(predictions_list)
        for index, row in df.iterrows():
            pred_item = next(pred_iter)
            assert isinstance(pred_item, dict)
            assert pred_item["rowId"] == index
            assert pred_item["prediction"] == row[0]
            assert isinstance(pred_item["predictionValues"], list)
            assert len(pred_item["predictionValues"]) == 1
            assert pred_item["predictionValues"][0]["label"] == config["target"]["name"]
            assert pred_item["predictionValues"][0]["value"] == row[0]

    def test_map_anomaly_prediction(self):
        d = {"Predictions": [1.2, 2.3, 3.4]}
        df = pd.DataFrame(data=d)
        config = parse_validate_deployment_config_file(self.deployment_config_anomaly)
        assert config["target"]["name"] == None
        assert config["target"]["type"] == "Anomaly"

        response = build_pps_response_json_str(df, config, TargetType.ANOMALY)
        response_json = json.loads(response)
        assert isinstance(response_json, dict)
        assert "data" in response_json
        predictions_list = response_json["data"]
        assert isinstance(predictions_list, list)
        assert len(predictions_list) == df.shape[0]

        pred_iter = iter(predictions_list)
        for index, row in df.iterrows():
            pred_item = next(pred_iter)
            print(pred_item)
            assert isinstance(pred_item, dict)
            assert pred_item["rowId"] == index
            assert pred_item["prediction"] == row[0]
            assert isinstance(pred_item["predictionValues"], list)
            assert len(pred_item["predictionValues"]) == 1
            assert pred_item["predictionValues"][0]["label"] == config["target"]["name"]
            assert pred_item["predictionValues"][0]["value"] == row[0]

    def test_map_binary_prediction(self):
        positive_class = "Iris-setosa"
        negative_class = "Iris-versicolor"
        d = {positive_class: [0.6, 0.5, 0.2], negative_class: [0.4, 0.5, 0.8]}
        df = pd.DataFrame(data=d)
        config = parse_validate_deployment_config_file(self.deployment_config_binary)
        assert config["target"]["name"] == "Species"
        assert config["target"]["type"] == "Binary"

        response = build_pps_response_json_str(df, config, TargetType.BINARY)
        response_json = json.loads(response)
        assert isinstance(response_json, dict)
        assert "data" in response_json
        predictions_list = response_json["data"]
        assert isinstance(predictions_list, list)
        assert len(predictions_list) == df.shape[0]

        pred_iter = iter(predictions_list)
        for index, row in df.iterrows():
            pred_item = next(pred_iter)
            assert isinstance(pred_item, dict)
            assert pred_item["rowId"] == index
            assert pred_item["predictionThreshold"] == config["target"]["prediction_threshold"]
            assert (
                pred_item["prediction"] == "Iris-setosa"
                if row[positive_class] > pred_item["predictionThreshold"]
                else negative_class
            )
            assert isinstance(pred_item["predictionValues"], list)
            assert len(pred_item["predictionValues"]) == 2

            # expected list must be formed in the [positive_class, negative_class] order
            # as that's how it is generated in map_binary_prediction
            assert pred_item["predictionValues"] == [
                {"label": positive_class, "value": row[positive_class]},
                {"label": negative_class, "value": 1 - row[positive_class]},
            ]

    def test_map_multiclass_prediction(self):
        class_labels = ["QSO", "STAR", "GALAXY"]
        d = {
            class_labels[0]: [0.6, 0.2, 0.3],
            class_labels[1]: [0.3, 0.4, 0.5],
            class_labels[2]: [0.1, 0.4, 0.2],
        }
        df = pd.DataFrame(data=d)
        config = parse_validate_deployment_config_file(self.deployment_config_multiclass)
        assert config["target"]["name"] == "class"
        assert config["target"]["type"] == "Multiclass"

        response = build_pps_response_json_str(df, config, TargetType.MULTICLASS)
        response_json = json.loads(response)
        assert isinstance(response_json, dict)
        assert "data" in response_json
        predictions_list = response_json["data"]
        assert isinstance(predictions_list, list)
        assert len(predictions_list) == df.shape[0]

        pred_iter = iter(predictions_list)
        expected_pred_iterator = iter(["QSO", "GALAXY", "STAR"])
        for index, row in df.iterrows():
            pred_item = next(pred_iter)

            assert isinstance(pred_item, dict)
            assert pred_item["rowId"] == index

            assert pred_item["prediction"] == next(expected_pred_iterator)
            assert isinstance(pred_item["predictionValues"], list)
            assert len(pred_item["predictionValues"]) == 3

            # expected list must be formed in the [GALAXY, QSO, STAR] order as classes are ordered this way
            # in map_multiclass_predictions according to the class mapping in deployment document
            assert pred_item["predictionValues"] == [
                {"label": class_labels[2], "value": row[class_labels[2]]},
                {"label": class_labels[0], "value": row[class_labels[0]]},
                {"label": class_labels[1], "value": row[class_labels[1]]},
            ]

    def test_map_text_generation_prediction(self):
        d = {"Predictions": ["Completion1", "Completion2", "Completion3"]}
        df = pd.DataFrame(data=d)
        config = parse_validate_deployment_config_file(self.deployment_config_text_generation)
        assert config["target"]["name"] == config["target"]["name"]
        assert config["target"]["type"] == "textgeneration"

        response = build_pps_response_json_str(df, config, TargetType.TEXT_GENERATION)
        response_json = json.loads(response)
        assert isinstance(response_json, dict)
        assert "data" in response_json
        predictions_list = response_json["data"]
        assert isinstance(predictions_list, list)
        assert len(predictions_list) == df.shape[0]

        pred_iter = iter(predictions_list)
        for index, row in df.iterrows():
            pred_item = next(pred_iter)
            print(pred_item)
            assert isinstance(pred_item, dict)
            assert pred_item["rowId"] == index
            assert pred_item["prediction"] == row[0]
            assert isinstance(pred_item["predictionValues"], list)
            assert len(pred_item["predictionValues"]) == 1
            assert pred_item["predictionValues"][0]["label"] == config["target"]["name"]
            assert pred_item["predictionValues"][0]["value"] == row[0]

    @pytest.mark.parametrize(
        "framework, problem, language, deployment_config",
        [
            (SKLEARN, REGRESSION, PYTHON, deployment_config_regression),
            (SKLEARN, BINARY, PYTHON, deployment_config_binary),
            (SKLEARN, MULTICLASS, PYTHON, deployment_config_multiclass),
        ],
    )
    @pytest.mark.parametrize("deployment_config_as_env_var", [True, False])
    def test_drum_prediction_server_pps_response(
        self,
        resources,
        framework,
        problem,
        language,
        deployment_config,
        deployment_config_as_env_var,
        tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
        )

        append_cmd = None
        if deployment_config_as_env_var:
            os.environ[ArgumentOptionsEnvVars.DEPLOYMENT_CONFIG] = deployment_config
        else:
            append_cmd = " --deployment-config {}".format(deployment_config)

        with DrumServerRun(
            resources.target_types(problem),
            resources.class_labels(framework, problem),
            custom_model_dir,
            append_cmd=append_cmd,
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
                    assert "data" in response_json
                    predictions_list = response_json["data"]
                    assert isinstance(predictions_list, list)
                    assert len(predictions_list)

                    prediction_item = predictions_list[0]
                    assert "rowId" in prediction_item
                    assert "prediction" in prediction_item
                    assert "predictionValues" in prediction_item

                    assert pd.read_csv(input_dataset).shape[0] == len(predictions_list)

        unset_drum_supported_env_vars()
