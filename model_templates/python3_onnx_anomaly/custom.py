"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from typing import List, Optional
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import OneClassSVM

#drum score --code-dir /Users/asli.demiroz/repos/datarobot-user-models/model_templates/python3_onnx_anomaly --target-type anomaly --input /Users/asli.demiroz/repos/datarobot-user-models/tests/testdata/juniors_3_year_stats_regression.csv

def transform(df, model):
    df = df.fillna(-99999)
    return df
