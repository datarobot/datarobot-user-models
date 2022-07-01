"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
from tempfile import NamedTemporaryFile
import os
import pandas as pd
import pytest
import requests

from datarobot_drum.drum.enum import EnvVarNames
from .constants import (
    BINARY,
    REGRESSION,
    RESPONSE_PREDICTIONS_KEY,
    UNSTRUCTURED,
)
from datarobot_drum.resource.drum_server_utils import DrumServerRun

from datarobot_drum.drum.utils.drum_utils import unset_drum_supported_env_vars


class TestInferenceCustomJavaPredictor:
    @pytest.mark.sequential
    @pytest.mark.parametrize(
        "problem, class_labels",
        [(REGRESSION, None), (BINARY, ["no", "yes"]), (UNSTRUCTURED, None),],
    )
    # current test case returns hardcoded predictions:
    # - for regression: [1, 2, .., N samples]
    # - for binary: [{0.3, 0,7}, {0.3, 0.7}, ...]
    # - for unstructured: "10"
    def test_custom_model_with_custom_java_predictor(
        self, resources, class_labels, problem,
    ):
        unset_drum_supported_env_vars()
        cur_file_dir = os.path.dirname(os.path.abspath(__file__))
        # have to point model dir to a folder with jar, so drum could detect the language
        model_dir = os.path.join(cur_file_dir, "custom_java_predictor")
        os.environ[
            EnvVarNames.DRUM_JAVA_CUSTOM_PREDICTOR_CLASS
        ] = "com.datarobot.test.TestCustomPredictor"
        os.environ[EnvVarNames.DRUM_JAVA_CUSTOM_CLASS_PATH] = os.path.join(model_dir, "*")
        with DrumServerRun(resources.target_types(problem), class_labels, model_dir,) as run:
            input_dataset = resources.datasets(None, problem)
            # do predictions
            post_args = {"data": open(input_dataset, "rb")}
            if problem == UNSTRUCTURED:
                response = requests.post(
                    run.url_server_address + "/predictUnstructured", **post_args
                )
            else:
                response = requests.post(run.url_server_address + "/predict", **post_args)
                print(response.text)
                assert response.ok
                predictions = json.loads(response.text)[RESPONSE_PREDICTIONS_KEY]
                actual_num_predictions = len(predictions)
                in_data = pd.read_csv(input_dataset)
                assert in_data.shape[0] == actual_num_predictions
            if problem == REGRESSION:
                assert list(range(1, actual_num_predictions + 1)) == predictions
            elif problem == UNSTRUCTURED:
                assert response.content.decode("UTF-8") == "10"
            else:
                single_prediction = {"yes": 0.7, "no": 0.3}
                assert [single_prediction] * actual_num_predictions == predictions

        unset_drum_supported_env_vars()
