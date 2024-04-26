from datarobot_drum import RuntimeParameters
from datarobot_drum.drum.gpu_predictors.utils import NGCRegistryClient


def load_model(code_dir: str):
    ngc_registry_url = RuntimeParameters.get("ngcRegistryUrl")
    ngc_credential = RuntimeParameters.get("ngcCredential")
    ngc_client = NGCRegistryClient(ngc_credential)
    ngc_client.download_model_version(ngc_registry_url)
    return "succeeded"  # a non-empty response is required to signal that load_model succeeded
