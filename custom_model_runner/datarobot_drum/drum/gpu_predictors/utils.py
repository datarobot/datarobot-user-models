from google.protobuf import text_format
from tritonclient.grpc.model_config_pb2 import ModelConfig

from datarobot_drum import RuntimeParameters
from datarobot_drum.drum.enum import TritonInferenceServerArtifacts
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.utils.drum_utils import DrumUtils


def read_model_config(model_repository_dir) -> ModelConfig:
    artifacts_found = DrumUtils.find_files_by_extensions(
        model_repository_dir, TritonInferenceServerArtifacts.ALL
    )
    if len(artifacts_found) == 0:
        raise DrumCommonException("No model configuration found, add a config.pbtxt")

    model_configs = []
    for artifact_file in artifacts_found:
        try:
            model_config = ModelConfig()
            with open(artifact_file, "r") as f:
                config_text = f.read()
                text_format.Merge(config_text, model_config)

            # skip ensemble model config
            if "ensemble" not in model_config.name:
                model_configs.append(model_config)

        except Exception as e:
            raise DrumCommonException(f"Can't read model configuration: {artifact_file}") from e

    if len(model_configs) > 1:
        raise DrumCommonException(
            "Found multiple model configurations. Multi-deployments are not supported yet."
        )

    return model_configs[0]


def get_optional_parameter(key, default_value=None):
    try:
        return RuntimeParameters.get(key)
    except ValueError:
        return default_value
