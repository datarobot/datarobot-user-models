"""
Copyright 2023 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
import os
from unittest.mock import patch

import pytest

from datarobot_drum.runtime_parameters.exceptions import InvalidJsonException
from datarobot_drum.runtime_parameters.exceptions import InvalidRuntimeParam
from datarobot_drum.runtime_parameters.runtime_parameters import RuntimeParameters
from datarobot_drum.runtime_parameters.runtime_parameters_schema import RuntimeParameterTypes


class TestRuntimeParameters:
    @pytest.mark.parametrize(
        "runtime_param_type, payload",
        [
            (RuntimeParameterTypes.STRING, "Some string value"),
            (
                RuntimeParameterTypes.CREDENTIAL,
                {
                    "credential_type": "s3",
                    "aws_access_key_id": "123aaa",
                    "aws_secret_access_key": "3425sdd",
                    "aws_session_token": "12345abcde",
                },
            ),
            (
                RuntimeParameterTypes.CREDENTIAL,
                {
                    "credential_type": "s3",
                    "region": "us-west",
                    "aws_access_key_id": "123aaa",
                    "aws_secret_access_key": "3425sdd",
                    "aws_session_token": "12345abcde",
                },
            ),
        ],
    )
    def test_valid(self, runtime_param_type, payload):
        runtime_param_name = "AAA"
        formatted_runtime_param_name = f"MLOPS_RUNTIME_PARAM_{runtime_param_name}"
        runtime_param_env_value = json.dumps({"type": runtime_param_type.value, "payload": payload})
        with patch.dict(os.environ, {formatted_runtime_param_name: runtime_param_env_value}):
            assert RuntimeParameters.get(runtime_param_name) == payload

    @pytest.mark.parametrize(
        "runtime_param_type, payload",
        [
            ("STRING", "Some string value"),
            ("str", "Some string value"),
            (
                "CREDENTIAL",
                {
                    "credential_type": "s3",
                    "aws_access_key_id": "123aaa",
                    "aws_secret_access_key": "3425sdd",
                    "aws_session_token": "12345abcde",
                },
            ),
            (
                "creds",
                {
                    "credential_type": "s3",
                    "aws_access_key_id": "123aaa",
                    "aws_secret_access_key": "3425sdd",
                    "aws_session_token": "12345abcde",
                },
            ),
        ],
    )
    def test_invalid_type(self, runtime_param_type, payload):
        self._read_runtime_param_and_expect_to_fail(runtime_param_type, payload)

    @staticmethod
    def _read_runtime_param_and_expect_to_fail(type, payload):
        runtime_param_name = "AAA"
        formatted_runtime_param_name = f"MLOPS_RUNTIME_PARAM_{runtime_param_name}"
        runtime_param_env_value = json.dumps({"type": type, "payload": payload})
        with patch.dict(os.environ, {formatted_runtime_param_name: runtime_param_env_value}):
            with pytest.raises(InvalidRuntimeParam) as ex:
                RuntimeParameters.get(runtime_param_name)
            assert "Invalid runtime parameter!" in str(ex.value)

    def test_invalid_aws_credential_type(self):
        payload = {
            "region": "us-west",
            "aws_access_key_id": "123aaa",  # Mutual exclusive
            "aws_secret_access_key": "3425sdd",  # Mutual exclusive
            "aws_session_token": "123aaa",  # Mutual exclusive
        }
        for invalid_credential_type in ("s4", "S3"):
            payload["credential_type"] = invalid_credential_type
            self._read_runtime_param_and_expect_to_fail(
                RuntimeParameterTypes.CREDENTIAL.value, payload
            )

    def test_missing_mandatory_aws_credential_attribute(self):
        payload = {
            "credential_type": "s3",
            "region": "us-west",
            "aws_access_key_id": "123aaa",  # Mutual exclusive
            "aws_secret_access_key": "3425sdd",  # Mutual exclusive
            "aws_session_token": "123aaa",  # Mutual exclusive
        }
        for missing_attr in (
            "credential_type",
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
        ):
            payload.pop(missing_attr)
            self._read_runtime_param_and_expect_to_fail(
                RuntimeParameterTypes.CREDENTIAL.value, payload
            )

    def test_empty_mandatory_aws_credential_attribute(self):
        payload = {
            "credential_type": "s3",
            "region": "us-west",
            "aws_access_key_id": "123aaa",  # Mutual exclusive
            "aws_secret_access_key": "3425sdd",  # Mutual exclusive
            "aws_session_token": "123aaa",  # Mutual exclusive
        }
        for missing_attr in (
            "credential_type",
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
        ):
            payload[missing_attr] = ""
            self._read_runtime_param_and_expect_to_fail(
                RuntimeParameterTypes.CREDENTIAL.value, payload
            )

    def test_invalid_json_env_value(self):
        runtime_param_name = "AAA"
        formatted_runtime_param_name = f"MLOPS_RUNTIME_PARAM_{runtime_param_name}"
        runtime_param_value = json.dumps(
            {"type": RuntimeParameterTypes.STRING.value, "payload": "abc"}
        )
        invalid_env_value = runtime_param_value + " - invalid"
        with patch.dict(os.environ, {formatted_runtime_param_name: invalid_env_value}):
            with pytest.raises(InvalidJsonException) as ex:
                RuntimeParameters.get(runtime_param_name)
            assert "Invalid runtime parameter json payload." in str(ex.value)
