#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#

import copy
import logging
import socket
from contextlib import closing

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor
from datarobot_drum.drum.language_predictors.python_predictor.python_predictor import (
    PythonPredictor,
)
from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.exceptions import DrumCommonException, DrumSerializationError
from datarobot_drum.drum.language_predictors.java_predictor.java_predictor import JavaPredictor
from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import PythonModelAdapter

logger = logging.getLogger(__name__)

try:
    from datarobot_drum.drum.language_predictors.r_predictor.r_predictor import RPredictor

    r_supported = True
except (ImportError, ModuleNotFoundError, DrumCommonException):
    r_supported = False


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
        {
            "positiveClassLabel": 1,
            "negativeClassLabel": 0,
            "target_type": TargetType.BINARY,
        },
        {
            "classLabels": ["a", "b", "c"],
            "target_type": TargetType.MULTICLASS,
        },
        {"target_type": TargetType.REGRESSION},
        {"target_type": TargetType.TEXT_GENERATION},
    ],
)
def test_lang_predictor_configure(predictor_params, essential_language_predictor_init_params):
    with patch(
        "datarobot_drum.drum.language_predictors.base_language_predictor."
        "read_model_metadata_yaml"
    ) as mock_read_model_metadata_yaml:
        mock_read_model_metadata_yaml.return_value = ""
        init_params = copy.deepcopy(essential_language_predictor_init_params)
        init_params.update(predictor_params)
        lang_predictor = FakeLanguagePredictor()
        lang_predictor.mlpiper_configure(init_params)
        if (
            predictor_params.get("positiveClassLabel") is not None
            and predictor_params.get("negativeClassLabel") is not None
        ):
            assert lang_predictor.positive_class_label == predictor_params["positiveClassLabel"]
            assert lang_predictor.negative_class_label == predictor_params["negativeClassLabel"]
            assert lang_predictor.class_labels is None
        elif predictor_params.get("classLabels"):
            assert lang_predictor.positive_class_label is None
            assert lang_predictor.negative_class_label is None
            assert lang_predictor.class_labels == ["a", "b", "c"]
        else:
            assert lang_predictor.positive_class_label is None
            assert lang_predictor.negative_class_label is None
            assert lang_predictor.class_labels is None

        mock_read_model_metadata_yaml.assert_called_once_with("custom_model_path")


class TestPythonPredictor(object):
    @pytest.mark.parametrize(
        "predictor_params, predictions, prediction_labels",
        [
            (
                {
                    "positiveClassLabel": 1,
                    "negativeClassLabel": 0,
                    "target_type": TargetType.BINARY,
                },
                np.array([[0.1, 0.9], [0.8, 0.2]]),
                [1, 0],
            ),
            (
                {
                    "classLabels": ["a", "b", "c"],
                    "target_type": TargetType.MULTICLASS,
                },
                np.array([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]]),
                ["a", "b", "c"],
            ),
            (
                {
                    "target_type": TargetType.REGRESSION,
                },
                np.array([1, 2]),
                None,
            ),
            (
                {
                    "target_type": TargetType.TEXT_GENERATION,
                },
                np.array(["a", "b"]),
                None,
            ),
        ],
    )
    def test_python_predictor_predict(
        self,
        predictor_params,
        predictions,
        prediction_labels,
        essential_language_predictor_init_params,
        mock_python_model_adapter_load_model_from_artifact,
        mock_python_model_adapter_predict,
    ):
        with patch(
            "datarobot_drum.drum.language_predictors.base_language_predictor."
            "read_model_metadata_yaml"
        ) as mock_read_model_metadata_yaml:
            mock_read_model_metadata_yaml.return_value = ""
            mock_python_model_adapter_predict.return_value = predictions, prediction_labels

            init_params = copy.deepcopy(essential_language_predictor_init_params)
            init_params.update(predictor_params)
            py_predictor = PythonPredictor()
            py_predictor.mlpiper_configure(init_params)
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
                    model=called_model,
                    target_type=predictor_params["target_type"],
                )

    def test_python_predictor_fails_to_load_artifact(
        self, essential_language_predictor_init_params
    ):
        """Ensure the model adapter raises a drum serialization error if it cannot load the artifact"""
        init_params = dict(
            essential_language_predictor_init_params, **{"target_type": TargetType.BINARY.value}
        )
        py_predictor = PythonPredictor()

        with pytest.raises(DrumSerializationError), patch.object(
            PythonModelAdapter, "load_model_from_artifact"
        ) as mock_load:
            mock_load.side_effect = Exception("artifact had an oops")
            py_predictor.mlpiper_configure(init_params)


@pytest.mark.skipif(not r_supported, reason="requires R framework to be installed")
@pytest.mark.parametrize("class_ordering", [lambda x: x, lambda x: list(reversed(x))])
class TestRPredictor(object):
    def test_r_predictor_replace_sanitized_class_names_same_binary(self, class_ordering):
        r_pred = RPredictor(positive_class_label="a", negative_class_label="b")
        predictions = pd.DataFrame(np.ones((3, 2)), columns=class_ordering(["a", "b"]))
        result = r_pred._replace_sanitized_class_names(predictions)
        assert list(result.columns) == class_ordering(["a", "b"])

    def test_r_predictor_replace_sanitized_class_names_unsanitary_binary(self, class_ordering):
        r_pred = RPredictor(positive_class_label="a+1", negative_class_label="b+1")
        predictions = pd.DataFrame(np.ones((3, 2)), columns=class_ordering(["a.1", "b.1"]))
        result = r_pred._replace_sanitized_class_names(predictions)
        assert list(result.columns) == class_ordering(["a+1", "b+1"])

    def test_r_predictor_replace_sanitized_class_names_float_binary(self, class_ordering):
        r_pred = RPredictor(positive_class_label="7.0", negative_class_label="7.1")
        predictions = pd.DataFrame(np.ones((3, 2)), columns=class_ordering(["X7", "X7.1"]))
        result = r_pred._replace_sanitized_class_names(predictions)
        assert list(result.columns) == class_ordering(["7.0", "7.1"])

    def test_r_predictor_replace_sanitized_class_names_same_multiclass(self, class_ordering):
        r_pred = RPredictor(class_labels=["a", "b", "c"])
        predictions = pd.DataFrame(np.ones((3, 3)), columns=class_ordering(["a", "b", "c"]))
        result = r_pred._replace_sanitized_class_names(predictions)
        assert list(result.columns) == class_ordering(["a", "b", "c"])

    def test_r_predictor_replace_sanitized_class_names_unsanitary_multiclass(self, class_ordering):
        r_pred = RPredictor(class_labels=["a+1", "b-1", "c$1"])
        predictions = pd.DataFrame(np.ones((3, 3)), columns=class_ordering(["a.1", "b.1", "c.1"]))
        result = r_pred._replace_sanitized_class_names(predictions)
        assert list(result.columns) == class_ordering(["a+1", "b-1", "c$1"])

    def test_r_predictor_replace_sanitized_class_names_float_multiclass(self, class_ordering):
        r_pred = RPredictor(class_labels=["7.0", "7.1", "7.2"])
        predictions = pd.DataFrame(np.ones((3, 3)), columns=class_ordering(["X7", "X7.1", "X7.2"]))
        result = r_pred._replace_sanitized_class_names(predictions)
        assert list(result.columns) == class_ordering(["7.0", "7.1", "7.2"])

    def test_r_predictor_replace_sanitized_class_names_ambiguous_multiclass(self, class_ordering):
        r_pred = RPredictor(class_labels=["a+1", "a-1", "a$1"])
        predictions = pd.DataFrame(np.ones((3, 3)), columns=class_ordering(["a.1", "a.1", "a.1"]))
        with pytest.raises(DrumCommonException, match="Class label names are ambiguous"):
            r_pred._replace_sanitized_class_names(predictions)


class TestJavaPredictor(object):
    # Verifying that correct code branch is taken depending on the data size.
    # As jp object is not properly configured, just check for the expected error message.
    @pytest.mark.parametrize(
        "data_size, error_message",
        [(2, "object has no attribute 'predict'"), (40000, "object has no attribute 'predictCSV'")],
    )
    def test_java_predictor_py4j_data(self, data_size, error_message):
        from datarobot_drum.drum.language_predictors.java_predictor.java_predictor import (
            JavaPredictor,
        )

        jp = JavaPredictor()
        with pytest.raises(AttributeError, match=error_message):
            jp._predict(binary_data=b"d" * data_size)

    @patch.object(JavaPredictor, "find_free_port")
    def test_run_java_server_entry_point_fail(self, mock_find_free_port):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            mock_find_free_port.return_value = s.getsockname()[1]

            pred = JavaPredictor()
            pred.model_artifact_extension = ".jar"

            # check that PredictorEntryPoint can not bind to port as it is taken
            with pytest.raises(DrumCommonException, match="java gateway failed to start"):
                pred._run_java_server_entry_point()

            # check that JavaGateway() fails to connect
            with pytest.raises(DrumCommonException, match="Failed to connect to java gateway"):
                pred._setup_py4j_client_connection()

    def test_run_java_server_entry_point_succeed(self):
        pred = JavaPredictor()
        pred.model_artifact_extension = ".jar"
        pred._run_java_server_entry_point()
        # required to properly shutdown py4j Gateway
        pred._setup_py4j_client_connection()
        pred._stop_py4j()
