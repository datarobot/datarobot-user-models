"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json

from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.enum import TargetType


def parse_validate_deployment_config_file(filename):
    if filename is None:
        return None
    with open(filename) as f:
        try:
            deployment_config = json.load(f)
            deployment_config["target"]
        except json.decoder.JSONDecodeError:
            raise DrumCommonException(
                "Failed to parse deployment config json file '{}'".format(filename)
            )
        except KeyError:
            raise DrumCommonException(
                "'target' section not found in deployment config json file {}".format(filename)
            )
    return deployment_config


def get_class_names_from_class_mapping(class_mapping):
    """
    Parameters
    ----------
    class_mapping : list(list), optional
        class_mapping as it represented in deployment_document.json:
        [["GALAXY", 0], ["QSO", 1], ["STAR", 2]]
    Returns
    -------
    classes, sorted list of class labels
    """
    if not class_mapping:
        return None
    return [kv[0] for kv in sorted(class_mapping, key=lambda v: v[1])]


def build_pps_response_json_str(out_data, deployment_config, target_type):
    target_info = deployment_config["target"]
    class_names_list = get_class_names_from_class_mapping(target_info["class_mapping"])
    data_lst = []
    if target_type == TargetType.MULTICLASS:
        f = map_multiclass_predictions
    elif target_type == TargetType.REGRESSION:
        f = map_regression_prediction
    elif target_type == TargetType.ANOMALY:
        target_info["name"] = "Anomaly Score"
        f = map_regression_prediction
    elif target_type == TargetType.BINARY:
        f = map_binary_prediction
    else:
        raise DrumCommonException("target type '{}' is not supported".format(target_type))

    for index, row in out_data.iterrows():
        data_lst.append(f(row, index, target_info, class_names_list))

    return json.dumps(dict(data=data_lst))


def map_regression_prediction(row, index, target_info, class_names):
    label = target_info["name"]
    pred_value = row.iloc[0]
    return {
        "prediction": pred_value,
        "predictionValues": [{"label": label, "value": pred_value}],
        "rowId": index,
    }


def map_multiclass_predictions(row, index, target_info, class_names):
    prediction_values = [
        {"label": class_name, "value": row[class_name]} for class_name in class_names
    ]
    decision = next(p_val for p_val in prediction_values if p_val["value"] >= max(row.values))[
        "label"
    ]
    return {"prediction": decision, "predictionValues": prediction_values, "rowId": index}


def map_binary_prediction(row, index, target_info, class_names):
    decision_threshold = target_info["prediction_threshold"]
    positive_class = class_names[1]
    negative_class = class_names[0]
    pred_value = row[positive_class]
    prediction_values = [
        {"label": positive_class, "value": pred_value},
        {"label": negative_class, "value": 1 - pred_value},
    ]
    decision = positive_class if pred_value > decision_threshold else negative_class
    return {
        "prediction": decision,
        "predictionValues": prediction_values,
        "predictionThreshold": decision_threshold,
        "rowId": index,
    }
