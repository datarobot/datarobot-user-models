"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json

import pandas as pd
from io import StringIO
from datarobot_drum.drum.language_predictors.java_predictor.java_predictor import JavaPredictor


class ScoringCodePredictor(JavaPredictor):
    def __init__(
        self,
        code_dir,
        target_type,
        negative_class_label=None,
        positive_class_label=None,
        class_labels=None,
        with_explanations=False,
    ):
        super(ScoringCodePredictor, self).__init__()
        self.with_explanations = with_explanations

        params = {
            "__custom_model_path__": code_dir,
            "target_type": target_type,
            "negativeClassLabel": negative_class_label,
            "positiveClassLabel": positive_class_label,
            "classLabels": class_labels,
            "withExplanations": with_explanations,
        }

        self.mlpiper_configure(params)

    def predict(self, df):
        """
            Score dataframe
            Parameters
            ----------
            df: DataFrame
                input data dataframe

            Returns
            -------
            predictions: DataFrame
                predictions dataframe
        """
        if self.with_explanations:
            raise Exception("Explanations are enabled, please use predict_with_explanations method")
        s = df.to_csv(index=False)
        bin_data = s.encode()
        return super(ScoringCodePredictor, self).predict(binary_data=bin_data)

    def predict_with_explanations(self, df):
        """
            Score dataframe with explanations
            Parameters
            ----------
            df: DataFrame
                input data dataframe

            Returns
            -------
            predictions: DataFrame
                predictions dataframe
            explanations: DataFrame
                explanations dataframe
        """
        if not self.with_explanations:
            raise Exception("Explanations are not enabled, please use predict method")
        s = df.to_csv(index=False)
        bin_data = s.encode()
        return super(ScoringCodePredictor, self).predict_with_explanations(binary_data=bin_data)


def load_model(code_dir):
    # Three predictors are loaded from 3 artifacts.
    # Multiclass predictor
    predictor1 = ScoringCodePredictor(
        "{}/skyserver_multiclass".format(code_dir), "multiclass", class_labels=["STAR", "GALAXY", "QSO"]
    )
    # Binary predictor.
    predictor2 = ScoringCodePredictor(
        "{}/skyserver_binary_star_galaxy".format(code_dir),
        "binary",
        positive_class_label="STAR",
        negative_class_label="GALAXY",
        with_explanations=True
    )
    # Regression predictor.
    predictor3 = ScoringCodePredictor(
        "{}/skyserver_reg".format(code_dir), "regression", with_explanations=True
    )
    return predictor1, predictor2, predictor3


def score(data, model, **kwargs):
    """
    The score() is required when running DRUM with `structured` target_type like regression/binary/multiclass

    """
    # Predict and print on multiclass predictor
    pred_df1 = model[0].predict(data)
    print(pred_df1)

    # Handle the failure with binary predictor
    try:
        pred_df2 = model[1].predict(data)
        print(pred_df2)
    except Exception as e:
        print(e)

    # Predict and print on regression predictor
    if model[2].with_explanations:
        # Predictions and explanations
        preds, expl = model[2].predict_with_explanations(data)
        print(preds)
        print(expl)
    else:
        preds = model[2].predict(data)
        print(preds)

    # return multiclass predictions as DRUM is started with multiclass parameters
    return pred_df1


def score_unstructured(model, data, query, **kwargs):
    """
        The score() is required when running DRUM with `structured` target_type like regression/binary/multiclass
    """

    # Unstructured input is text. Make a DF for compatibility
    data = pd.read_csv(StringIO(data))

    # Predict and print on multiclass predictor.
    # MULTICLASS Scoring code doesn't support predictions explanations yet.
    print("PREDICTING MULTICLASS")
    pred_df1 = model[0].predict(data)
    print(pred_df1)

    print("PREDICTING BINARY")
    # Handle the failure with binary predictor
    try:
        if model[1].with_explanations:
            # Predictions and explanations
            preds, expl = model[1].predict_with_explanations(data)
            ret_obj = {}
            ret_obj["predictions"] = preds.to_dict("records")
            ret_obj["explanations"] = expl.to_dict("records")
            ret_json_str_1 = json.dumps(ret_obj)
        else:
            preds = model[1].predict(data)
            ret_obj = preds.to_dict("records")
            ret_json_str_1 = json.dumps(ret_obj)
    except Exception as e:
        print(e)

    print("PREDICTING REGRESSION")
    # Predict and print on regression predictor
    if model[2].with_explanations:
        # Predictions and explanations
        preds, expl = model[2].predict_with_explanations(data)
        ret_obj = preds.to_dict("list")
        ret_obj["explanations"] = expl.to_dict("records")
        ret_json_str_2 = json.dumps(ret_obj)
    else:
        preds = model[2].predict(data)
        ret_obj = preds.to_dict("list")
        ret_json_str_2 = json.dumps(ret_obj)
        #print(preds)

    # # return multiclass predictions as DRUM is started with multiclass parameters
    return ret_json_str_2
