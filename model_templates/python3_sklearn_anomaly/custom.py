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


def transform(df, model):
    df = df.fillna(-99999)
    return df
