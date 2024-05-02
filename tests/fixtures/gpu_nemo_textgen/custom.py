from datarobot_drum import RuntimeParameters
from datarobot_drum.drum.gpu_predictors.utils import S3Client


def load_model(code_dir: str):
    s3_url = RuntimeParameters.get("s3Url")

    s3_credential = RuntimeParameters.get("s3Credential")
    s3_client = S3Client(s3_url, s3_credential)

    keys, dirs = s3_client.list_objects()
    s3_client.download_files(keys, dirs)
    return "succeeded"  # a non-empty response is required to signal that load_model succeeded
