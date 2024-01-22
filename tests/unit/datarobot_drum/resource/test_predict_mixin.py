#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
from unittest.mock import Mock

import pandas as pd

from datarobot_drum.drum.enum import PRED_COLUMN
from datarobot_drum.resource.predict_mixin import PredictMixin


class TestPredictionResponse:
    """Tests the prediction's response format"""

    def test_regression_prediction_response(self):
        prediction_response = Mock(
            predictions=pd.DataFrame({PRED_COLUMN: [0.1, 0.2]}), extra_model_output=None
        )
        response = PredictMixin._build_drum_response_json_str(prediction_response)
        assert response == '{"predictions":[0.1,0.2]}'

    def test_regression_prediction_response_with_extra_model_output(self):
        prediction_response = Mock(
            predictions=pd.DataFrame({PRED_COLUMN: [0.1, 0.2]}),
            extra_model_output=pd.DataFrame({"extra1": [2, 3], "extra2": ["high", "low"]}),
        )
        response = PredictMixin._build_drum_response_json_str(prediction_response)
        assert (
            response
            == '{"predictions":[0.1,0.2],"extraModelOutput":{"columns":["extra1","extra2"],"index":[0,1],"data":[[2,"high"],[3,"low"]]}}'
        )

    def test_binary_prediction_response(self):
        prediction_response = Mock(
            predictions=pd.DataFrame({"0": [0.1, 0.2], "1": [0.9, 0.8]}), extra_model_output=None
        )
        response = PredictMixin._build_drum_response_json_str(prediction_response)
        assert response == '{"predictions":[{"0":0.1,"1":0.9},{"0":0.2,"1":0.8}]}'

    def test_binary_prediction_response_with_extra_model_output(self):
        prediction_response = Mock(
            predictions=pd.DataFrame({"0": [0.1, 0.2], "1": [0.9, 0.8]}),
            extra_model_output=pd.DataFrame({"extra1": [2, 3], "extra2": ["high", "low"]}),
        )
        response = PredictMixin._build_drum_response_json_str(prediction_response)
        assert (
            response
            == '{"predictions":[{"0":0.1,"1":0.9},{"0":0.2,"1":0.8}],"extraModelOutput":{"columns":["extra1","extra2"],"index":[0,1],"data":[[2,"high"],[3,"low"]]}}'
        )

    def test_multiclass_prediction_response(self):
        prediction_response = Mock(
            predictions=pd.DataFrame({"cat": [0.1], "dog": [0.7], "horse": [0.2]}),
            extra_model_output=None,
        )
        response = PredictMixin._build_drum_response_json_str(prediction_response)
        assert response == '{"predictions":[{"cat":0.1,"dog":0.7,"horse":0.2}]}'

    def test_multiclass_prediction_response_with_extra_model_output(self):
        prediction_response = Mock(
            predictions=pd.DataFrame({"cat": [0.1], "dog": [0.7], "horse": [0.2]}),
            extra_model_output=pd.DataFrame({"extra1": [2, 3], "extra2": ["high", "low"]}),
        )
        response = PredictMixin._build_drum_response_json_str(prediction_response)
        assert (
            response
            == '{"predictions":[{"cat":0.1,"dog":0.7,"horse":0.2}],"extraModelOutput":{"columns":["extra1","extra2"],"index":[0,1],"data":[[2,"high"],[3,"low"]]}}'
        )
