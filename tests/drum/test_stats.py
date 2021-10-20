import json
from tempfile import NamedTemporaryFile
import pandas as pd
import pytest
import requests

from datarobot_drum.drum.enum import ModelInfoKeys, TargetType
from datarobot_drum.drum.description import version as drum_version
from .constants import (
    DOCKER_PYTHON_SKLEARN,
    PYTHON,
    REGRESSION,
    RESPONSE_PREDICTIONS_KEY,
    SKLEARN,
)
from datarobot_drum.resource.drum_server_utils import DrumServerRun
from datarobot_drum.resource.utils import _create_custom_model_dir

from datarobot_drum.drum.utils import unset_drum_supported_env_vars


class TestInference:
    @pytest.fixture
    def temp_file(self):
        with NamedTemporaryFile() as f:
            yield f

    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            # (SKLEARN, REGRESSION, PYTHON, DOCKER_PYTHON_SKLEARN),
            (SKLEARN, REGRESSION, PYTHON, None),
        ],
    )
    def test_custom_models_with_drum_prediction_server(
        self, resources, framework, problem, language, docker, tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources, tmp_path, framework, problem, language,
        )

        unset_drum_supported_env_vars()
        with DrumServerRun(
            resources.target_types(problem),
            resources.class_labels(framework, problem),
            custom_model_dir,
            docker,
        ) as run:
            input_dataset = resources.datasets(framework, problem)
            # do predictions
            for endpoint in ["/predict/", "/predictions/"]:
                for post_args in [
                    {"files": {"X": open(input_dataset)}},
                    {"data": open(input_dataset, "rb")},
                ]:
                    response = requests.post(run.url_server_address + endpoint, **post_args)

                    print(response.text)
                    assert response.ok
                    actual_num_predictions = len(
                        json.loads(response.text)[RESPONSE_PREDICTIONS_KEY]
                    )
                    in_data = pd.read_csv(input_dataset)
                    assert in_data.shape[0] == actual_num_predictions
            # test model info
            response = requests.get(run.url_server_address + "/stats/")

            assert response.ok
            stats = response.json()
            sections = ["drum_info", "mem_info", "time_info"]
            assert all([s in stats for s in sections])
            mem_info = stats["mem_info"]
            assert mem_info["drum_rss"] > 0
            assert mem_info["free"] > 0
            assert mem_info["total"] > 0
            if docker is not None:
                assert mem_info["container_limit"] > 0
                assert mem_info["container_max_used"] > 0
                assert mem_info["container_used"] > 0

        unset_drum_supported_env_vars()
