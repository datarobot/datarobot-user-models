"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

from datarobot_drum.drum.language_predictors.java_predictor.java_predictor import JavaPredictor


class ScoringCodePredictor(JavaPredictor):
    def __init__(
        self,
        code_dir,
        target_type,
        negative_class_label=None,
        positive_class_label=None,
        class_labels=None,
    ):
        super(ScoringCodePredictor, self).__init__()

        params = {
            "__custom_model_path__": code_dir,
            "target_type": target_type,
            "negativeClassLabel": negative_class_label,
            "positiveClassLabel": positive_class_label,
            "classLabels": class_labels,
        }

        self.mlpiper_configure(params)

    def predict(self, df):
        s = df.to_csv(index=False)
        bin_data = s.encode()
        return super(ScoringCodePredictor, self).predict(binary_data=bin_data)


def load_model(code_dir):
    # Three predictors are loaded from 2 artifacts.
    # Multiclass predictor
    predictor1 = ScoringCodePredictor(
        "{}/skyserver_class".format(code_dir), "multiclass", class_labels=["STAR", "GALAXY", "QSO"]
    )
    # Binary predictor, which fails as predictions won't sum up to 1. Keep it to show the error handling.
    predictor2 = ScoringCodePredictor(
        "{}/skyserver_class".format(code_dir),
        "binary",
        positive_class_label="STAR",
        negative_class_label="GALAXY",
    )
    # Regression predictor.
    predictor3 = ScoringCodePredictor(
        "{}/skyserver_reg".format(code_dir),
        "regression",
    )
    return predictor1, predictor2, predictor3


def score(data, model, **kwargs):
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
    pred_df3 = model[2].predict(data)
    print(pred_df3)

    # return multiclass predictions as DRUM is started with multiclass parameters
    return pred_df1
