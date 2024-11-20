#
#  Copyright 2024 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
import os
import tempfile

import pytest
import yaml

from datarobot_drum.drum.lazy_loading.constants import BackendType
from datarobot_drum.drum.lazy_loading.constants import LazyLoadingEnvVars
from datarobot_drum.drum.lazy_loading.lazy_loading_handler import LazyLoadingHandler
from datarobot_drum.drum.lazy_loading.schema import LazyLoadingCommandLineFileCredentialsContent
from datarobot_drum.drum.lazy_loading.schema import LazyLoadingCommandLineFileContent
from datarobot_drum.drum.lazy_loading.schema import LazyLoadingFile
from datarobot_drum.drum.lazy_loading.schema import LazyLoadingRepository


class TestSetupEnvironmentVariablesFromValuesFile:
    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def command_line_lazy_loading_file_content(self):
        return LazyLoadingCommandLineFileContent(
            files=[
                LazyLoadingFile(
                    remote_path="remote/path/a",
                    local_path="a",
                    repository_id="1",
                ),
                LazyLoadingFile(
                    remote_path="remote/path/b",
                    local_path="b",
                    repository_id="1",
                ),
                LazyLoadingFile(
                    remote_path="remote/path/c",
                    local_path="c",
                    repository_id="2",
                ),
            ],
            repositories=[
                LazyLoadingRepository(
                    repository_id="1",
                    bucket_name="minio-bucket",
                    credential_id="100",
                    endpoint_url="https://minio.local:9000",
                    verify_certificate=False,
                ),
                LazyLoadingRepository(
                    repository_id="2",
                    bucket_name="s3-bucket1",
                    credential_id="101",
                ),
            ],
            credentials=[
                LazyLoadingCommandLineFileCredentialsContent(
                    id="100",
                    credential_type=BackendType.S3.value,
                    aws_access_key_id="user",
                    aws_secret_access_key="1234qwer",
                ),
                LazyLoadingCommandLineFileCredentialsContent(
                    id="101",
                    credential_type=BackendType.S3.value,
                    aws_access_key_id="ASIASRCPMEEETYKXLMOM",
                    aws_secret_access_key="MILSKFTx26eieFnvWjzUnI3j/V+jwLwJc6hTpUIq",
                    aws_session_token="iIQoJb3JpZ2luX2VjEB8aCXVzLWVhc3QtMSJGMEQCIGMUdK5KPeSMApHi",
                ),
            ],
        )

    @pytest.fixture
    def command_line_lazy_loading_filepath(self, tmp_dir, command_line_lazy_loading_file_content):
        filepath = f"{tmp_dir}/lazy_loading.yaml"
        with open(filepath, "w") as file:
            model_dict = command_line_lazy_loading_file_content.model_dump(
                mode="json", by_alias=True
            )
            yaml.dump(model_dict, file)
            yield file.name

    def test_success(
        self, command_line_lazy_loading_filepath, command_line_lazy_loading_file_content
    ):
        os.environ.pop(LazyLoadingEnvVars.get_lazy_loading_data_key(), None)

        credential_env_keys = {}
        credential_prefix = LazyLoadingEnvVars.get_repository_credential_id_key_prefix()
        for creds_entry in command_line_lazy_loading_file_content.credentials:
            credential_env_key = f"{credential_prefix}_{creds_entry.id.upper()}"
            credential_env_keys[creds_entry.id] = (credential_env_key, creds_entry)
            os.environ.pop(credential_env_key, None)

        LazyLoadingHandler.setup_environment_variables_from_values_file(
            command_line_lazy_loading_filepath
        )
        expected_lazy_loading_json = command_line_lazy_loading_file_content.model_dump_json(
            exclude={"credentials"}
        )
        actual_lazy_loading_json = os.environ[LazyLoadingEnvVars.get_lazy_loading_data_key()]
        assert actual_lazy_loading_json == expected_lazy_loading_json
        for creds_id, (credentials_env_key, creds_entry) in credential_env_keys.items():
            expected_credentials_json = creds_entry.model_dump_json(exclude={"id"})
            actual_credentials_json = os.environ[credentials_env_key]
            assert actual_credentials_json == expected_credentials_json

    def test_invalid_lazy_loading_yaml_content(self, command_line_lazy_loading_filepath):
        new_filepath_with_incorrect_content = command_line_lazy_loading_filepath.replace(
            ".yaml", "_new.yaml"
        )
        with open(command_line_lazy_loading_filepath, "r") as infile, open(
            new_filepath_with_incorrect_content, "w"
        ) as outfile:
            next(infile)
            for line in infile.readlines():
                outfile.write(line)
        with pytest.raises(ValueError, match="Invalid lazy loading YAML file content."):
            LazyLoadingHandler.setup_environment_variables_from_values_file(
                new_filepath_with_incorrect_content
            )

    @pytest.mark.parametrize("section_name", ["files", "repositories", "credentials"])
    def test_missing_required_section(self, section_name, command_line_lazy_loading_filepath):
        new_filepath_with_incorrect_content = command_line_lazy_loading_filepath.replace(
            ".yaml", "_new.yaml"
        )
        with open(command_line_lazy_loading_filepath, "r") as infile, open(
            new_filepath_with_incorrect_content, "w"
        ) as outfile:
            for line in infile.readlines():
                if line.startswith(section_name):
                    line = line.replace(section_name, f"{section_name}Other")
                outfile.write(line)
        with pytest.raises(ValueError, match="Invalid lazy loading content."):
            LazyLoadingHandler.setup_environment_variables_from_values_file(
                new_filepath_with_incorrect_content
            )

    @pytest.mark.parametrize(
        "field_name",
        [
            "localPath",
            "remotePath",
            "repositoryId",
            "bucketName",
            "credentialId",
            "awsAccessKeyId",
            "awsSecretAccessKey",
            "credentialType",
            "id",
        ],
    )
    def test_missing_required_field(self, field_name, command_line_lazy_loading_filepath):
        new_filepath_with_incorrect_content = command_line_lazy_loading_filepath.replace(
            ".yaml", "_new.yaml"
        )
        with open(command_line_lazy_loading_filepath, "r") as infile, open(
            new_filepath_with_incorrect_content, "w"
        ) as outfile:
            for line in infile.readlines():
                if field_name in line:
                    line = line.replace(field_name, f"{field_name}Other")
                outfile.write(line)
        with pytest.raises(ValueError, match="Invalid lazy loading content."):
            LazyLoadingHandler.setup_environment_variables_from_values_file(
                new_filepath_with_incorrect_content
            )
