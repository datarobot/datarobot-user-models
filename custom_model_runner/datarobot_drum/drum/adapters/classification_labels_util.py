"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import sys

import numpy as np
import pandas as pd

from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.exceptions import DrumCommonException


def needs_class_labels(
    target_type, negative_class_label=None, positive_class_label=None, class_labels=None
):
    if target_type == TargetType.BINARY:
        return negative_class_label is None or positive_class_label is None
    if target_type == TargetType.MULTICLASS:
        return class_labels is None
    return False


def possibly_intuit_order(
    input_filename, target_type, target_filename=None, target_name=None,
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
    if target_type.value in TargetType.CLASSIFICATION.value:
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
                    len(possible_class_labels), possible_class_labels,
                )
            )

    return possible_class_labels
