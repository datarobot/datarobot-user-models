"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import os
import pickle
import pandas as pd
import onnxruntime

preprocessor = None

# drum score --code-dir /Users/asli.demiroz/repos/datarobot-user-models/model_templates/python3_onnx_multiclass --input /Users/asli.demiroz/repos/datarobot-user-models/tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv --class-labels-file /Users/asli.demiroz/repos/datarobot-user-models/model_templates/python3_pytorch_multiclass/class_labels.txt --target-type multiclass

def load_model(code_dir: str):
    """
    Can be used to load supported models if your model has multiple artifacts, or for loading
    models that DRUM  does not natively support

    Parameters
    ----------
    code_dir : is the directory where model artifact and additional code are provided, passed in

    Returns
    -------
    If used, this hook must return a non-None value
    """
    global preprocessor
    with open(os.path.join(code_dir, "preprocessor.pkl"), mode="rb") as f:
        preprocessor = pickle.load(f)

    import onnxruntime
    ort_session = onnxruntime.InferenceSession(os.path.join(code_dir, "multiclass_SDSS.onnx"))
    return ort_session


def transform(data: pd.DataFrame, model) -> pd.DataFrame:
    """
    Intended to apply transformations to the prediction data before making predictions. This is
    most useful if DRUM supports the model's library, but your model requires additional data
    processing before it can make predictions

    Parameters
    ----------
    data : is the dataframe given to DRUM to make predictions on
    model : is the deserialized model loaded by DRUM or by `load_model`, if supplied

    Returns
    -------
    Transformed data
    """
    if preprocessor is None:
        raise ValueError("Preprocessor not loaded")

    return pd.DataFrame(preprocessor.transform(data))
