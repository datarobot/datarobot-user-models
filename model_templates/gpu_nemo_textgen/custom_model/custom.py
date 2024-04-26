import os
import boto3
from jinja2 import Environment, FileSystemLoader
from mlpiper.extra.aws_helper import AwsHelper
from pathlib import Path

from datarobot_drum import RuntimeParameters
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX
from custom_model_runner.datarobot_drum.custom_task_interfaces.user_secrets import (
    ApiTokenSecret,
    SecretType,
)


# this path is required by NeMo Inference Server
NEMO_MODEL_STORE_DIR = "/model-store"


def load_model(code_dir: str):
    s3_url = RuntimeParameters.get("s3Url")
    s3_credential = RuntimeParameters.get("s3Credential")
    s3_client = S3Client(s3_url, s3_credential)

    keys, dirs = s3_client.list_objects()
    s3_client.download_files(NEMO_MODEL_STORE_DIR, keys, dirs)
    return "success"  # a non empty response is required to signal that model is loaded successfully


class NGCRegistryClient:
    NGC_CONFIG_TEMPLATE = "ngc_config_template.j2"
    logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

    def __init__(self):
        templates_dir = os.path.dirname(__file__)
        file_loader = FileSystemLoader(templates_dir)
        self.env = Environment(file_loader)

    def set_configuration(self):
        template = self.env.get_template(self.NGC_CONFIG_TEMPLATE)
        cred_dict = RuntimeParameters.get("ngcCredential")
        assert (
            cred_dict["credential_type"] == SecretType.API_TOKEN.value
        ), "NGC credential should be of type API_TOKEN"
        ngc_credential = ApiTokenSecret.from_dict(cred_dict)

        ngc_config = template.render(
            ngc_api_key=ngc_credential.api_token,
            ngc_org=RuntimeParameters.get("ngcOrg"),
            ngc_team=RuntimeParameters.get("ngcTeam"),
        )

        # create ~/.ngc/config
        user_home_path = os.environ["HOME"]
        ngc_config_dir = f"{user_home_path}/.ngc"
        ngc_config_file = f"{ngc_config_dir}/config"

        if not Path(ngc_config_file).exists():
            self.logger.info("Creating a new NGC configuration")
            Path().mkdir(exist_ok=True)
            with open(ngc_config_file, "w") as f:
                f.write(ngc_config)
        else:
            self.logger.info("Found existing NGC configuration. Reusing existing one.")

    def pull_model(self, model_name_and_tag):
        cmd = ["ngc", "registry", "model", "download-version", model_name_and_tag]
        p = subprocess.Popen(
            cmd,
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = p.communicate()

        if len(stdout):
            self.logger.info(stdout)
        if len(stderr):
            self.logger.error(stderr)

        assert p.returncode == 0, "Failed to download model version from NGC registry"


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
