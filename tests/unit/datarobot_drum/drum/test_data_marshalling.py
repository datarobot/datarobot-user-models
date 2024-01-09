"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from datarobot_drum.drum.artifact_predictors.sklearn_predictor import SKLearnPredictor
from datarobot_drum.drum.data_marshalling import (
    _marshal_labels,
    _order_by_float,
    marshal_predictions,
)
from datarobot_drum.drum.enum import EXTRA_MODEL_OUTPUT_COLUMN, TargetType, PRED_COLUMN
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import PythonModelAdapter


def test_marshal_labels():
    assert _marshal_labels(request_labels=["True", "False"], model_labels=[False, True]) == [
        "False",
        "True",
    ]


def test__order_by_float():
    assert _order_by_float(["0", "01"], ["1.0", ".0"]) == ["01", "0"]
    assert _order_by_float(["0", "1"], [1.0, 0.0]) == ["1", "0"]
    assert _order_by_float(["0", "1"], ["1.0", "0.0"]) == ["1", "0"]
    assert _order_by_float(["0.0", "1"], ["1", ".0"]) == ["1", "0.0"]
    assert _order_by_float(["1.0", "2.4", "0.4", "1.4"], [2.4, 1.0, 0.4, 1.4]) == [
        "2.4",
        "1.0",
        "0.4",
        "1.4",
    ]


def test_marshal_predictions_multiclass_happy():
    preds = np.array([[1, 0, 0], [1, 0, 0]])
    labels = [1, 2, 3]
    expected = pd.DataFrame(data=preds, columns=labels)
    pd.testing.assert_frame_equal(
        marshal_predictions(
            request_labels=labels, predictions=preds, target_type=TargetType.MULTICLASS
        ),
        expected,
    )


def test_marshal_predictions_multiclass_wrong_label_length():
    preds = np.array([[1, 0, 0], [1, 0, 0]])
    labels = [1, 2, 3, 4]
    with pytest.raises(
        DrumCommonException,
        match=" predictions must return the probability distribution for the correct number of class labels",
    ):
        marshal_predictions(
            request_labels=labels, predictions=preds, target_type=TargetType.MULTICLASS
        )


def test_marshal_predictions_multiclass_wrong_label_length_cols_greater():
    preds = np.array([[1, 0, 0], [1, 0, 0]])
    labels = [1, 2]
    with pytest.raises(
        DrumCommonException,
        match=" predictions must return the probability distribution for the correct number of class labels",
    ):
        marshal_predictions(
            request_labels=labels, predictions=preds, target_type=TargetType.MULTICLASS
        )


def test_marshal_predictions_binary_reshape_happy():
    preds = np.array([1, 1, 1])
    labels = [1, 2]
    preds = marshal_predictions(
        request_labels=labels, predictions=preds, target_type=TargetType.BINARY
    )
    assert preds.equals(pd.DataFrame({1: [0, 0, 0], 2: [1, 1, 1]}))


def test_marshal_predictions_binary_both_rows_happy():
    preds = np.array([[0, 1], [0, 1], [0, 1]])
    labels = [1, 2]
    preds = marshal_predictions(
        request_labels=labels, predictions=preds, target_type=TargetType.BINARY
    )
    assert preds.equals(pd.DataFrame({1: [0, 0, 0], 2: [1, 1, 1]}))


def test_marshal_predictions_reshape_regression_happy():
    preds = np.array([1, 1, 1])
    labels = [1]
    res = marshal_predictions(
        request_labels=labels, predictions=preds, target_type=TargetType.REGRESSION
    )
    assert res.equals(pd.DataFrame({PRED_COLUMN: preds}))


def test_marshal_predictions_invalid_dtype():
    preds = np.zeros((1, 2, 3))
    labels = [1]
    with pytest.raises(
        DrumCommonException, match="predictions must return a np array, but received"
    ):
        marshal_predictions(
            request_labels=labels, predictions="elephant", target_type=TargetType.REGRESSION
        )
    with pytest.raises(DrumCommonException, match="predictions must return a 2 dimensional array"):
        marshal_predictions(
            request_labels=labels, predictions=preds, target_type=TargetType.REGRESSION
        )


def test_marshal_predictions_bad_shape_regression():
    labels = [1]
    preds = np.zeros((1, 2))
    with pytest.raises(DrumCommonException, match="must contain only 1 column"):
        marshal_predictions(
            request_labels=labels, predictions=preds, target_type=TargetType.REGRESSION
        )


def test_marshal_predictions_reshape_text_generation_happy():
    preds = np.array(["a", "b", "c"])
    labels = [PRED_COLUMN]
    res = marshal_predictions(
        request_labels=labels, predictions=preds, target_type=TargetType.TEXT_GENERATION
    )
    assert res.equals(pd.DataFrame({PRED_COLUMN: preds}))


def test_marshal_predictions_text_generation_externa_model_output():
    preds = np.array([["pred_1", "extra_1"], ["pred_2", "extra_2"]])

    labels = None
    res = marshal_predictions(
        request_labels=labels, predictions=preds, target_type=TargetType.TEXT_GENERATION
    )
    assert res.equals(
        pd.DataFrame(
            {PRED_COLUMN: ["pred_1", "pred_2"], EXTRA_MODEL_OUTPUT_COLUMN: ["extra_1", "extra_2"]}
        )
    )


def test_marshal_predictions_text_generation_invalid_dtype():
    # A (2,2,2) predictions are not valid text gen predictions
    preds = np.array([[["a", "b"], ["c", "d"]], [["e", "f"], ["g", "h"]]])
    labels = [PRED_COLUMN]
    with pytest.raises(
        DrumCommonException, match="predictions must return a np array, but received"
    ):
        marshal_predictions(
            request_labels=labels, predictions="a", target_type=TargetType.TEXT_GENERATION
        )
    with pytest.raises(DrumCommonException, match="predictions must return a 2 dimensional array"):
        marshal_predictions(
            request_labels=labels, predictions=preds, target_type=TargetType.TEXT_GENERATION
        )

    preds = np.array([["pred_1", "metadata_1", "extra_col"], ["pred_2", "metadata_2", "extra_col"]])
    with pytest.raises(
        DrumCommonException,
        match="Text Generation must contain 1 column with predictions or 2 columns with predictions and extra model output",
    ):
        marshal_predictions(
            request_labels=labels, predictions=preds, target_type=TargetType.TEXT_GENERATION
        )


def test_marshal_predictions_bad_shape_regression():
    preds = np.array([["a", "c", "d"], ["p", "q", "r"]])
    labels = [PRED_COLUMN]
    with pytest.raises(DrumCommonException, match="must contain only 1 column"):
        marshal_predictions(
            request_labels=labels, predictions=preds, target_type=TargetType.REGRESSION
        )


def test_marshal_predictions_dont_add_to_one():
    labels = [1, 2]
    preds = np.zeros((2, 2))
    with pytest.raises(DrumCommonException, match="prediction probabilities do not add up to 1"):
        marshal_predictions(
            request_labels=labels, predictions=preds, target_type=TargetType.MULTICLASS
        )


def test_marshal_predictions_add_to_one_weird():
    labels = [1, 2]
    preds = np.array([[-2, 3], [1, 0]])
    with pytest.raises(
        DrumCommonException, match="Your prediction probabilities have negative values"
    ):
        marshal_predictions(request_labels=labels, predictions=preds, target_type=TargetType.BINARY)


@pytest.mark.parametrize("data_dtype,label_dtype", [(int, int), (float, int), (int, float)])
def test_sklearn_predictor_wrong_dtype_labels(data_dtype, label_dtype):
    """
    This test makes sure that the target values can be ints, and the class labels be floats, and
    everything still works okay.

    TODO: Remove model adapter portion and only test _marshal_labels
    """
    X = pd.DataFrame({"col1": range(10)})
    y = pd.Series(data=[data_dtype(0)] * 5 + [data_dtype(1)] * 5)
    csv_bytes = bytes(X.to_csv(index=False), encoding="utf-8")
    estimator = LogisticRegression()
    estimator.fit(X, y)
    adapter = PythonModelAdapter(model_dir=None, target_type=TargetType.BINARY)
    adapter._predictor_to_use = SKLearnPredictor()
    preds, cols = adapter.predict(
        estimator,
        positive_class_label=str(label_dtype(0)),
        negative_class_label=str(label_dtype(1)),
        binary_data=csv_bytes,
        target_type=TargetType.BINARY,
    )
    marshalled_cols = _marshal_labels(
        request_labels=[str(label_dtype(1)), str(label_dtype(0))],
        model_labels=cols,
    )
    assert marshalled_cols == [str(label_dtype(0)), str(label_dtype(1))]
