"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import numpy as np
import pandas as pd
import pytest


from datarobot_drum.drum.data_marshalling import (
    _marshal_labels,
    _order_by_float,
    marshal_predictions,
)
from datarobot_drum.drum.enum import TargetType, REGRESSION_PRED_COLUMN
from datarobot_drum.drum.exceptions import DrumCommonException


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
    assert res.equals(pd.DataFrame({REGRESSION_PRED_COLUMN: preds}))


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
