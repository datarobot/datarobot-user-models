"""
Unit tests for vLLM Dockerfile validation.

This module validates the vLLM dropin environment Dockerfile configuration without requiring
GPU hardware. It performs static analysis to catch configuration errors
(e.g., WITH_ERROR_SERVER=0s instead of 0) before deployment.

Test Coverage:
    - Environment variable type and value validation
    - Typo detection in variable values
    - Boolean flag format validation
    - Path variable format validation

For detailed methodology and validation rule derivation, see vllm_validator.py
"""

import os
import pytest
from tests.unit.dockerfile_validators.vllm_validator import VllmDockerfileValidator


class TestVllmDockerfileStaticValidation:
    """Static validation tests for vLLM Dockerfile (no Docker build required)."""

    @pytest.fixture
    def vllm_dockerfile_path(self):
        """Provide the path to the vLLM Dockerfile."""
        return os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "../../../public_dropin_gpu_environments/vllm/Dockerfile"
            )
        )

    @pytest.fixture
    def validator(self, vllm_dockerfile_path):
        """Provide vLLM Dockerfile validator instance."""
        return VllmDockerfileValidator(vllm_dockerfile_path)

    def test_dockerfile_exists(self, vllm_dockerfile_path):
        """Verify that the vLLM Dockerfile exists and is readable."""
        assert os.path.exists(
            vllm_dockerfile_path
        ), f"vLLM Dockerfile not found at {vllm_dockerfile_path}"
        assert os.path.isfile(
            vllm_dockerfile_path
        ), f"Path exists but is not a file: {vllm_dockerfile_path}"

    def test_env_vars_validation(self, validator):
        """
        Validate all environment variables have correct types and values.
        """
        validation_errors = validator.validate_env_vars()

        assert not validation_errors, (
            f"\nEnvironment variable validation failed with {len(validation_errors)} error(s):"
            + "".join(validation_errors)
        )

    def test_no_typos_in_env_values(self, validator):
        """
        Detect common typos in environment variable values.

        Common errors caught:
            - '0s' instead of '0'
            - Leading or trailing whitespace
            - Unexpected suffixes or prefixes
        """
        typo_errors = validator.check_typos()
        assert (
            not typo_errors
        ), f"\nTypo validation failed with {len(typo_errors)} error(s):" + "".join(typo_errors)

