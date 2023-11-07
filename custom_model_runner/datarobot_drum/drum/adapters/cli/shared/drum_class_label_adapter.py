"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import sys

import numpy as np
import pandas as pd
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX
from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.exceptions import DrumCommonException

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


def needs_class_labels(
    target_type, negative_class_label=None, positive_class_label=None, class_labels=None
):
    if target_type == TargetType.BINARY:
        return negative_class_label is None or positive_class_label is None
    if target_type == TargetType.MULTICLASS:
        return class_labels is None
    return False


def possibly_intuit_order(
    input_filename,
    target_type,
    target_filename=None,
    target_name=None,
):
    if target_type == TargetType.ANOMALY:
        return None
    elif target_filename:
        assert target_name is None

        y = pd.read_csv(target_filename, index_col=False)
        classes = np.unique(y.iloc[:, 0].astype(str))
    else:
        assert target_filename is None
        df = pd.read_csv(input_filename)
        if target_name not in df.columns:
            e = "The column '{}' does not exist in your dataframe. \nThe columns in your dataframe are these: {}".format(
                target_name, list(df.columns)
            )
            print(e, file=sys.stderr)
            raise DrumCommonException(e)
        uniq = df[target_name].astype(str).unique()
        classes = set(uniq) - {np.nan}
    if len(classes) >= 2:
        return sorted(classes)
    elif len(classes) == 1:
        raise DrumCommonException("Only one target label was provided, please revise training data")
    return None


def infer_class_labels(target_type, input_filename, target_filename=None, target_name=None):
    # No class label information was supplied, but we may be able to infer the labels
    if target_type.is_classification():
        print("WARNING: class list not supplied. Using unique target values.")

    possible_class_labels = possibly_intuit_order(
        input_filename=input_filename,
        target_filename=target_filename,
        target_name=target_name,
        target_type=target_type,
    )

    if possible_class_labels is None:
        raise DrumCommonException(
            "Target type {} requires class label information. No labels were supplied and "
            "labels could not be inferred from the target.".format(target_type.value)
        )

    if target_type == TargetType.BINARY:
        if len(possible_class_labels) != 2:
            raise DrumCommonException(
                "Target type binary requires exactly 2 class labels. Detected {}: {}".format(
                    len(possible_class_labels), possible_class_labels
                )
            )

    elif target_type == TargetType.MULTICLASS:
        if len(possible_class_labels) < 2:
            raise DrumCommonException(
                "Target type multiclass requires more than 2 class labels. Detected {}: {}".format(
                    len(possible_class_labels),
                    possible_class_labels,
                )
            )

    return possible_class_labels


class DrumClassLabelAdapter(object):
    """
    Shared parent class for other adapters that handles parsing class label input parameters.
    """

    def __init__(
        self,
        target_type,
        positive_class_label=None,
        negative_class_label=None,
        class_labels=None,
    ):
        """
        Parameters
        ----------
        target_type: datarobot_drum.drum.enum.TargetType
        positive_class_label: str or None
            Optional. Name of the positive class label if target type is binary
        negative_class_label: str or None
            Optional. Name of the negative class label if target type is binary
        class_labels: list[str] or None
            Optional. List of class labels
        """
        self.target_type = target_type
        self.positive_class_label = positive_class_label
        self.negative_class_label = negative_class_label
        self.class_labels = class_labels

    @property
    def class_ordering(self):
        if self.negative_class_label is not None and self.positive_class_label is not None:
            class_order = [self.positive_class_label, self.negative_class_label]
        elif self.class_labels:
            class_order = self.class_labels
        else:
            class_order = None

        return class_order

    def _infer_class_labels_if_not_provided(
        self, input_filename, target_filename=None, target_name=None
    ):
        if needs_class_labels(
            target_type=self.target_type,
            negative_class_label=self.negative_class_label,
            positive_class_label=self.positive_class_label,
            class_labels=self.class_labels,
        ):
            # TODO: Only pass y in here as a dataframe
            class_labels = infer_class_labels(
                target_type=self.target_type,
                input_filename=input_filename,
                target_filename=target_filename,
                target_name=target_name,
            )

            if self.target_type == TargetType.BINARY:
                self.positive_class_label, self.negative_class_label = class_labels
            elif self.target_type == TargetType.MULTICLASS:
                self.class_labels = class_labels
