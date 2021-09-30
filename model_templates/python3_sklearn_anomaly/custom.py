from typing import List, Optional
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import OneClassSVM


def transform(df, model):
    df = df.fillna(-99999)
    return df
