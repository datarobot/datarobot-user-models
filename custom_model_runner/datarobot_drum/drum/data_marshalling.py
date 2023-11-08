"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
from distutils.util import strtobool
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from datarobot_drum.drum.common import TargetType
from datarobot_drum.drum.enum import (
    LOGGER_NAME_PREFIX,
    PRED_COLUMN,
)
from datarobot_drum.drum.exceptions import DrumCommonException

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


def marshal_predictions(
    request_labels: Union[List[str], None],
    predictions: np.array,
    target_type: TargetType,
    model_labels: Optional[List[Any]] = None,
) -> pd.DataFrame:
    """
    :param request_labels: These labels are sent in from DataRobot in every prediction request
    :param predictions: The predictions returned from the custom code
    :param target_type: Which sort of target the model is supposed to work on
    :param model_labels: Optionally, what labels the model thinks the predictions should use
    :return:
    Predictions DataFrame
    """
    predictions = _validate_dimensionality_and_type(predictions)
    if target_type.is_classification():
        return _classification_marshal_preds(predictions, request_labels, model_labels)
    elif target_type.is_single_column():
        return _single_col_marshal_preds(predictions)
    return predictions


def _marshal_labels(request_labels: List[str], model_labels: List[Any]):
    if model_labels is not None:
        if set(_standardize(r_l) for r_l in request_labels) == set(
            _standardize(m_l) for m_l in model_labels
        ):
            return _order_by_float(request_labels, model_labels)

        raise DrumCommonException(
            "Expected predictions to have columns {}, but encountered {}".format(
                request_labels, model_labels
            )
        )
    return request_labels


def _order_by_float(expected_labels, actual_labels):
    """
    Match the order of actual labels to the values in expected labels
    Given both can be cast to floats
    >>> _order_by_float(["1.0", "2.4", "0.4", "1.4"],  [2.4, 1.0, 0.4, 1.4])
    ['2.4', '1.0', '0.4', '1.4']
    """

    def get_corresponding_expected_label(a_l):
        for e_l in expected_labels:
            if _standardize(a_l) == _standardize(e_l):
                return e_l

    return [get_corresponding_expected_label(_l) for _l in actual_labels]


def _standardize(label):
    # First, hope it's a float
    try:
        return float(label)
    except ValueError:
        pass

    # Maybe if its a boolean we can make it floaty anyways
    try:
        return float(strtobool(label))
    except ValueError:
        pass

    # Okay lets just do a str.lower
    assert isinstance(label, str)
    return label.lower()


def _classification_marshal_preds(predictions, request_labels, model_labels):
    request_labels = _marshal_labels(request_labels, model_labels)
    predictions = _infer_negative_class_probabilities(predictions, request_labels)
    _validate_amount_of_columns(request_labels, predictions)
    _validate_probabilities_sum_to_one(predictions)
    return pd.DataFrame(predictions, columns=request_labels)


def _single_col_marshal_preds(predictions):
    _validate_predictions_are_one_dimensional(predictions)
    return pd.DataFrame(predictions, columns=[PRED_COLUMN])


def _validate_dimensionality_and_type(predictions):
    if not isinstance(predictions, np.ndarray):
        raise DrumCommonException(
            f"predictions must return a np array, but received {type(predictions)}"
        )
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)
    if len(predictions.shape) != 2:
        raise DrumCommonException(
            f"predictions must return a 2 dimensional array, but we received a {len(predictions.shape)} dimensional array"
        )
    return predictions


def _validate_predictions_are_one_dimensional(predictions):
    if predictions.shape[1] != 1:
        raise DrumCommonException(
            f"Regression, Text Generation and anomaly predictions must contain only 1 column. "
            f"Your predictions have {predictions.shape[1]} columns"
        )


def _validate_probabilities_sum_to_one(predictions):
    if (predictions < 0).any():
        raise DrumCommonException("Your prediction probabilities have negative values")
    if (predictions > 1).any():
        raise DrumCommonException("Your prediction probabilities have values greater than 1")
    added_probs = predictions.sum(axis=1)
    good_preds = np.isclose(added_probs, 1)
    if not np.all(good_preds):
        bad_rows = predictions[~good_preds]
        raise DrumCommonException(
            "Your prediction probabilities do not add up to 1. \n{}".format(bad_rows)
        )


def _validate_amount_of_columns(labels_to_use, predictions):
    if predictions.shape[1] != len(labels_to_use):
        raise DrumCommonException(
            "Your predictions must return the "
            "probability distribution for the correct number of class labels. "
            "Expected {} columns, but received {}".format(len(labels_to_use), predictions.shape[1])
        )


def _infer_negative_class_probabilities(predictions, labels):
    if predictions.shape[1] == 1:
        if len(labels) == 1:
            raise DrumCommonException(
                f"Classification problems must have more than one class label"
            )
        if len(labels) != 2:
            raise DrumCommonException(
                f"Only one dimension of probability returned, but there are {len(labels)} class labels in this model"
            )
        neg_label = labels[0]
        pos_label = labels[1]
        pred_df = pd.DataFrame(predictions, columns=[pos_label])
        pred_df[neg_label] = 1 - pred_df[pos_label]
        return pred_df[[neg_label, pos_label]].values
    return predictions


def get_request_labels(
    class_labels: List[str],
    positive_class_label: str,
    negative_class_label: str,
) -> List[str]:
    """This function returns the requested class labels for both binary classification and multi-classification cases.

    Parameters
    ----------
    class_labels:
        Class labels. It is part of the parameters to initialize BaseLanguagePredictor.
    positive_class_label
        Positive class label. It is one init parameter of BaseLanguagePredictor.
    negative_class_label
        Negative class label. It is one init parameter of BaseLanguagePredictor.

    Returns
    -------
    Classification labels.
    """
    return class_labels if class_labels else [negative_class_label, positive_class_label]
