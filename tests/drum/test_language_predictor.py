"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import copy

import numpy as np
import pytest
from unittest.mock import patch

from custom_model_runner.datarobot_drum.drum.language_predictors.base_language_predictor import (
    BaseLanguagePredictor,
)
from custom_model_runner.datarobot_drum.drum.language_predictors.python_predictor.python_predictor import (
    PythonPredictor,
)
from datarobot_drum.drum.enum import TargetType


class FakeLanguagePredictor(BaseLanguagePredictor):
    def _predict(self, **kwargs):
        pass

    def _transform(self, **kwargs):
        pass

    def has_read_input_data_hook(self):
        pass


@pytest.mark.parametrize(
    "predictor_params",
    [
        {"positiveClassLabel": 1, "negativeClassLabel": 0, "target_type": TargetType.BINARY,},
        {"classLabels": ["a", "b", "c"], "target_type": TargetType.MULTICLASS,},
        {"target_type": TargetType.REGRESSION},
    ],
)
def test_lang_predictor_configure(predictor_params, essential_language_predictor_init_params):
    with patch(
        "custom_model_runner.datarobot_drum.drum.language_predictors.base_language_predictor."
        "read_model_metadata_yaml"
    ) as mock_read_model_metadata_yaml:
        mock_read_model_metadata_yaml.return_value = ""
        init_params = copy.deepcopy(essential_language_predictor_init_params)
        init_params.update(predictor_params)
        lang_predictor = FakeLanguagePredictor()
        lang_predictor.configure(init_params)
        if (
            predictor_params.get("positiveClassLabel") is not None
            and predictor_params.get("negativeClassLabel") is not None
        ):
            assert lang_predictor._positive_class_label == predictor_params["positiveClassLabel"]
            assert lang_predictor._negative_class_label == predictor_params["negativeClassLabel"]
            assert lang_predictor._class_labels is None
        elif predictor_params.get("classLabels"):
            assert lang_predictor._positive_class_label is None
            assert lang_predictor._negative_class_label is None
            assert lang_predictor._class_labels == ["a", "b", "c"]
        else:
            assert lang_predictor._positive_class_label is None
            assert lang_predictor._negative_class_label is None
            assert lang_predictor._class_labels is None

        mock_read_model_metadata_yaml.assert_called_once_with("custom_model_path")


@pytest.mark.parametrize(
    "predictor_params, predictions, prediction_labels",
    [
        (
            {"positiveClassLabel": 1, "negativeClassLabel": 0, "target_type": TargetType.BINARY,},
            np.array([[0.1, 0.9], [0.8, 0.2]]),
            [1, 0],
        ),
        (
            {"classLabels": ["a", "b", "c"], "target_type": TargetType.MULTICLASS,},
            np.array([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]]),
            ["a", "b", "c"],
        ),
        ({"target_type": TargetType.REGRESSION,}, np.array([1, 2]), None),
    ],
)
def test_python_predictor_predict(
    predictor_params,
    predictions,
    prediction_labels,
    essential_language_predictor_init_params,
    mock_python_model_adapter_load_model_from_artifact,
    mock_python_model_adapter_predict,
):
    with patch(
        "custom_model_runner.datarobot_drum.drum.language_predictors.base_language_predictor."
        "read_model_metadata_yaml"
    ) as mock_read_model_metadata_yaml:
        mock_read_model_metadata_yaml.return_value = ""
        mock_python_model_adapter_predict.return_value = predictions, prediction_labels

        init_params = copy.deepcopy(essential_language_predictor_init_params)
        init_params.update(predictor_params)
        py_predictor = PythonPredictor()
        py_predictor.configure(init_params)
        pred_params = {"target_type": predictor_params["target_type"]}
        py_predictor.predict(**pred_params)

        called_model = mock_python_model_adapter_load_model_from_artifact.return_value
        if (
            predictor_params.get("positiveClassLabel") is not None
            and predictor_params.get("negativeClassLabel") is not None
        ):
            mock_python_model_adapter_predict.assert_called_once_with(
                model=called_model,
                negative_class_label=predictor_params["negativeClassLabel"],
                positive_class_label=predictor_params["positiveClassLabel"],
                target_type=predictor_params["target_type"],
            )
        elif predictor_params.get("classLabels"):
            mock_python_model_adapter_predict.assert_called_once_with(
                model=called_model,
                class_labels=predictor_params["classLabels"],
                target_type=predictor_params["target_type"],
            )
        else:
            mock_python_model_adapter_predict.assert_called_once_with(
                model=called_model, target_type=predictor_params["target_type"],
            )
