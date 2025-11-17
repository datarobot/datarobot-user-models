"""
Unit tests for vLLM Dockerfile validation.

This module validates the vLLM dropin environment Dockerfile configuration without requiring
GPU hardware. It performs static analysis and optional Docker build tests to catch
configuration errors (e.g., WITH_ERROR_SERVER=0s instead of 0) before deployment.
** Testing should be extended for built image
Test Coverage:
    1. Static Analysis:
       - Environment variable type and value validation
       - Typo detection in variable values
       - Boolean flag format validation
       - Path variable format validation

    2. Docker Build (Optional):
       - Tests if image can be built successfully
       - Validates basic Python installation
       - Does not require GPU for build

For detailed methodology and validation rule derivation, see vllm_validator.py
"""

import os
import pytest
from tests.unit.dockerfile_validators.vllm_validator import VllmDockerfileValidator
from tests.unit.dockerfile_validators.docker_build_tester import DockerBuildTester


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


class TestVllmDockerfileBuild:
    """Docker build tests for vLLM Dockerfile."""

    @pytest.fixture(scope="class")
    def build_tester(self, request):
        dockerfile_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../public_dropin_gpu_environments/vllm")
        )
        image_tag = "test-vllm-image:latest"
        tester = DockerBuildTester(dockerfile_dir, image_tag)
        request.addfinalizer(tester.cleanup)
        return tester

    def test_dockerfile_builds_successfully(self, build_tester):
        """
        Test that the Dockerfile can be built into a Docker image.
        """
        success, output, elapsed_time = build_tester.build_image()
        assert success, f"Docker build failed after {elapsed_time:.1f}s:\n{output[-1000:]}"

    def test_python_installation(self, build_tester):
        """
        Test that Python is installed in the built image.
        """
        success, output = build_tester.test_basic_command(["python", "--version"])
        assert success, f"Python not found in built image:\n{output}"

    def test_cleanup(self, build_tester):
        """
        Ensure the test image is cleaned up after tests.
        """
        assert build_tester.cleanup(), "Failed to clean up Docker image."
