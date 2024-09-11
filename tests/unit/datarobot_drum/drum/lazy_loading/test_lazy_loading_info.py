import json
import os
import shutil
from pathlib import Path

import pytest
from datarobot_drum.lazy_loading.constants import MLOPS_LAZY_LOADING_DATA_ENV_VARIABLE

from custom_model_runner.datarobot_drum.lazy_loading.constants import MLOPS_REPOSITORY_SECRET_PREFIX
from custom_model_runner.datarobot_drum.lazy_loading.remote_files import RemoteFiles

@pytest.fixture
def code_root_dir():
    return "/tmp/code"

# Fetch credentials form Env variables and create single runtime
# env variable MLOPS_REPOSITORY_SECRET to emulate DRUM env
def set_aws_credentials(
        credential_id,
        mode="regular",
        cred_runtime_param_name=None,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        aws_session_token=None,
        endpoint_url=None,
):
    credential_payload = {
        "credentialType": "s3",
        "awsAccessKeyId": aws_access_key_id,
        "awsSecretAccessKey": aws_secret_access_key,
    }

    if endpoint_url is not None:
        credential_payload["endpointUrl"] = endpoint_url

    if aws_session_token is not None:
        credential_payload["sessionToken"] = aws_session_token

    mlops_credential_secret = {"type": "credential", "payload": credential_payload}
    os.environ[cred_runtime_param_name] = json.dumps(mlops_credential_secret)

@pytest.fixture
def get_credential():
    credential_id = "669d2623485e94b838e637bb"
    return credential_id

@pytest.fixture
def lazy_loading_config(get_credential):

    lazy_loading_config_dict = {
        "repositories": [
            {
                "type": "s3",
                "repository_id": "669d2623485e94b838e637bf",
                "bucket_name": "llm-artifacts-dev",
                "credential_id": get_credential
            },
        ],
        "files": [
            {
                "remote_path": "llm-artifacts/artifact_1.bin",
                "local_path": "/tmp/artifact_1.bin",
                "repository_id": "669d2623485e94b838e637bf"
            },
            {
                "remote_path": "llm-artifacts/artifact_2.bin",
                "local_path": "/tmp/artifact_2.bin",
                "repository_id": "669d2623485e94b838e637bf"
            },
        ]
    }
    return lazy_loading_config_dict


@pytest.fixture
def set_lazy_loading_env_config(lazy_loading_config, get_credential):

    # Fetch credentials form Env variables and create single runtime
    # env variable MLOPS_REPOSITORY_SECRET to emulate DRUM env
    # TODO: Credentials storing logic not finalized yet
    mlops_credential_env_variable = MLOPS_REPOSITORY_SECRET_PREFIX + get_credential.upper()
    set_aws_credentials(
        credential_id=get_credential,
        mode="runtime",
        cred_runtime_param_name=mlops_credential_env_variable,
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        endpoint_url=(
            os.environ['STORAGE_ENDPOINT_URL'] if 'STORAGE_ENDPOINT_URL' in os.environ else None
        ),
        aws_session_token=(
            os.environ['AWS_SESSION_TOKEN'] if 'AWS_SESSION_TOKEN' in os.environ else None
        ),
    )
    os.environ[MLOPS_LAZY_LOADING_DATA_ENV_VARIABLE] = json.dumps(lazy_loading_config)


class TestLazyLoadingConfig(object):

    def test_model_downloader(self, code_root_dir, lazy_loading_config, set_lazy_loading_env_config):

        remote_files = RemoteFiles(local_dir=code_root_dir).from_env_config()

        for remote_file in remote_files.get_remote_files():
            # TODO: add assert statements
            assert remote_file.repository_id == lazy_loading_config["repositories"][0]["repository_id"]


