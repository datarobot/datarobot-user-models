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
    predictor1 = ScoringCodePredictor("{}/junior_reg_1".format(code_dir), "regression")
    predictor2 = ScoringCodePredictor("{}/junior_reg_2".format(code_dir), "regression")
    return predictor1, predictor2


def score(data, model, **kwargs):
    pred_df1 = model[0].predict(data)
    pred_df2 = model[1].predict(data)

    if pred_df1.iloc[0]["Predictions"] > pred_df2.iloc[0]["Predictions"]:
        return pred_df1
    else:
        return pred_df2
