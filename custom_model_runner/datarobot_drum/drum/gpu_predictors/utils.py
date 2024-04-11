from google.protobuf import text_format
from tritonclient.grpc.model_config_pb2 import ModelConfig

from datarobot_drum.drum.enum import TritonInferenceServerArtifacts
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.utils.drum_utils import DrumUtils


def read_model_config(model_repository_dir) -> ModelConfig:
    artifacts_found = DrumUtils.find_files_by_extensions(
        model_repository_dir,
        TritonInferenceServerArtifacts.ALL
    )
    if len(artifacts_found) == 0:
        raise DrumCommonException("No model configuration found, add a config.pbtxt")
    elif len(artifacts_found) > 1:
        raise DrumCommonException(
            "Found multiple model configurations. Multi-deployments are not supported yet."
        )

    model_config_pbtxt = artifacts_found[0]

    try:
        model_config = ModelConfig()
        with open(model_config_pbtxt, "r") as f:
            config_text = f.read()
            text_format.Merge(config_text, model_config)

        return model_config
    except Exception as e:
        raise DrumCommonException(
            f"Can't read model configuration: {model_config_pbtxt}"
        ) from e
