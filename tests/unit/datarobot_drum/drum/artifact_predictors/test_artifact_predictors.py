"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

from datarobot_drum.drum.artifact_predictors.keras_predictor import KerasPredictor
from datarobot_drum.drum.artifact_predictors.onnx_predictor import ONNXPredictor
from datarobot_drum.drum.artifact_predictors.pmml_predictor import PMMLPredictor
from datarobot_drum.drum.artifact_predictors.sklearn_predictor import SKLearnPredictor
from datarobot_drum.drum.artifact_predictors.torch_predictor import PyTorchPredictor
from datarobot_drum.drum.artifact_predictors.xgboost_predictor import XGBoostPredictor


def test_artifact_predictor_extension():
    assert SKLearnPredictor().is_artifact_supported("artifact.PKL")
    assert XGBoostPredictor().is_artifact_supported("artifact.PkL")
    assert PyTorchPredictor().is_artifact_supported("artifact.pTh")
    assert KerasPredictor().is_artifact_supported("artifact.h5")
    assert PMMLPredictor().is_artifact_supported("artifact.PmMl")
    assert ONNXPredictor().is_artifact_supported("artifact.onnx")
    assert not SKLearnPredictor().is_artifact_supported("artifact.jar")
    assert not XGBoostPredictor().is_artifact_supported("artifact.jar")
    assert not PyTorchPredictor().is_artifact_supported("artifact.Jar")
    assert not KerasPredictor().is_artifact_supported("artifact.jaR")
    assert not PMMLPredictor().is_artifact_supported("artifact.jAr")
    assert not ONNXPredictor().is_artifact_supported("artifact.JAR")
