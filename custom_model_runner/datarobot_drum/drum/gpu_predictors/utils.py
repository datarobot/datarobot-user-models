import logging
import os
import subprocess
from pathlib import Path
from subprocess import CalledProcessError

from datarobot_drum.drum.enum import TritonInferenceServerArtifacts
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.utils.drum_utils import DrumUtils

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)
MODEL_STORE_DIR = "/model-store"


def read_model_config(model_repository_dir):
    from google.protobuf import text_format
    from tritonclient.grpc.model_config_pb2 import ModelConfig

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


class NGCRegistryClient:
    def __init__(self, ngc_credential):
        self.api_token = ngc_credential["apiToken"]
        self._configure_ngc_client()

    def download_model_version(self, ngc_registry_url, destination_dir=MODEL_STORE_DIR):
        ngc_org, ngc_team = self._parse_ngc_registry_url(ngc_registry_url)

        cmd = [
            "ngc",
            "registry",
            "model",
            "download-version",
            "--org",
            ngc_org,
            "--team",
            ngc_team,
            "--dest",
            destination_dir,
            ngc_registry_url,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
            )
            logger.info(result.stdout)
        except CalledProcessError as e:
            logger.error(e.stderr)
            raise DrumCommonException(
                f"Failed to download a model version from the NGC registry:\n {ngc_registry_url}"
            )

    @staticmethod
    def _parse_ngc_registry_url(ngc_registry_url):
        url_parts = ngc_registry_url.split("/")
        if len(url_parts) != 3:
            raise ValueError(
                "Invalid NGC registry URL format. Expected format: {org}/{team}/{model_name}:{tag}"
            )

        # org, team
        return url_parts[0], url_parts[1]

    def _configure_ngc_client(self):
        # the NGC configuration is expected to be at ~/.ngc/config
        user_home_path = os.environ["HOME"]
        self.ngc_config_dir = f"{user_home_path}/.ngc"
        self.ngc_config_file = f"{self.ngc_config_dir}/config"
        ngc_config = f"[CURRENT]\napikey = {self.api_token}"

        Path(self.ngc_config_dir).mkdir(exist_ok=True)
        with open(self.ngc_config_file, "w") as f:
            f.write(ngc_config)


class S3Client:
    def __init__(self, s3_url, credential):
        import boto3
        from mlpiper.extra.aws_helper import AwsHelper

        parsed_url = AwsHelper.s3_url_parse(s3_url)
        self.bucket_name = parsed_url[0]
        self.prefix = parsed_url[1]

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=credential["awsAccessKeyId"],
            aws_secret_access_key=credential["awsSecretAccessKey"],
            aws_session_token=credential["awsSessionToken"]
            if "awsSessionToken" in credential
            else None,
        )

    def list_objects(self):
        keys = []
        dirs = []
        next_token = ""
        base_kwargs = {
            "Bucket": self.bucket_name,
            "Prefix": self.prefix,
        }
        while next_token is not None:
            kwargs = base_kwargs.copy()
            if next_token != "":
                kwargs.update({"ContinuationToken": next_token})
            results = self.s3_client.list_objects_v2(**kwargs)
            contents = results.get("Contents")
            for i in contents:
                k = i.get("Key")
                if k[-1] != "/":
                    keys.append(k)
                else:
                    dirs.append(k)
            next_token = results.get("NextContinuationToken")

        return keys, dirs

    def download_files(self, keys, dirs, destination_dir=MODEL_STORE_DIR):
        for d in dirs:
            dest_pathname = os.path.join(destination_dir, str(d).replace(self.prefix, ""))
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
        for k in keys:
            dest_pathname = os.path.join(destination_dir, str(k).replace(self.prefix, ""))
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            self.s3_client.download_file(self.bucket_name, k, dest_pathname)
