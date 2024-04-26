import logging
import os
import subprocess
from pathlib import Path
from subprocess import CalledProcessError

import boto3
from mlpiper.extra.aws_helper import AwsHelper

from datarobot_drum import RuntimeParameters
from datarobot_drum.custom_task_interfaces.user_secrets import SecretType
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.gpu_predictors import get_optional_parameter

# this path is required by NeMo Inference Server
NEMO_MODEL_STORE_DIR = "/model-store"


def load_model(code_dir: str):
    load_model_response = (
        "succeeded"  # a non-empty response is required to signal that load_model succeeded
    )

    s3_url = get_optional_parameter("s3Url")
    if s3_url:
        s3_credential = RuntimeParameters.get("s3Credential")
        s3_client = S3Client(s3_url, s3_credential)

        keys, dirs = s3_client.list_objects()
        s3_client.download_files(NEMO_MODEL_STORE_DIR, keys, dirs)
        return load_model_response

    ngc_registry_url = get_optional_parameter("ngcRegistryUrl")
    if ngc_registry_url:
        ngc_credential = RuntimeParameters.get("ngcCredential")
        ngc_client = NGCRegistryClient(ngc_credential)
        ngc_client.download_model_version(ngc_registry_url)
        return load_model_response

    raise Exception(
        "Can't download model artifacts. The 'ngcRegistryUrl' or 's3Url'"
        " are expected to be set in the Runtime Parameters."
    )


class NGCRegistryClient:
    logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

    def __init__(self, ngc_credential):
        if ngc_credential["credentialType"] != SecretType.API_TOKEN.value:
            raise ValueError("NGC credential is expected to be of type 'API_TOKEN'.")
        self.api_token = ngc_credential["apiToken"]
        self._configure_ngc_client()

    def download_model_version(self, ngc_registry_url):
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
            ngc_registry_url,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
            )
        except CalledProcessError as e:
            self.logger.error(e.stderr)
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

    def download_files(self, destination_dir, keys, dirs):
        for d in dirs:
            dest_pathname = os.path.join(destination_dir, str(d).replace(self.prefix, ""))
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
        for k in keys:
            dest_pathname = os.path.join(destination_dir, str(k).replace(self.prefix, ""))
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            self.s3_client.download_file(self.bucket_name, k, dest_pathname)
