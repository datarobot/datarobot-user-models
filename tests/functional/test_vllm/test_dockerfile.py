"""
Copyright 2025 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.

vLLM Dockerfile build and validation tests.

These tests validate that the vLLM Docker image can be built successfully
and contains all required components for running in production.
"""
import os
import time

import docker
import pytest

from tests.constants import PUBLIC_DROPIN_GPU_ENVS_PATH


class TestVLLMDockerBuild:
    """Docker build tests for vLLM environment (requires Docker)."""

    VLLM_PATH = os.path.join(PUBLIC_DROPIN_GPU_ENVS_PATH, "vllm")
    IMAGE_TAG = "vllm-test:latest"

    @pytest.fixture(scope="class")
    def docker_client(self):
        """Provides a Docker client, failing if Docker is not available."""
        try:
            client = docker.from_env()
            client.ping()
            return client
        except Exception as e:
            pytest.fail(f"Docker is not available or accessible: {e}")

    @pytest.fixture(scope="class")
    def vllm_image(self, docker_client):
        """Builds the vLLM Docker image once per test class."""
        try:
            image, build_logs = docker_client.images.build(
                path=self.VLLM_PATH,
                tag=self.IMAGE_TAG,
                rm=True,
                forcerm=True,
            )
        except docker.errors.BuildError as e:
            build_log = "".join([log.get("stream", "") for log in e.build_log or []])
            pytest.fail(
                f"Docker build failed: {e}\nBuild Log (last 50 lines):\n{build_log[-5000:]}"
            )

        yield image

        # Cleanup: remove the image after tests
        try:
            docker_client.images.remove(image.id, force=True)
        except docker.errors.APIError:
            pass  # Ignore errors during cleanup

    @pytest.fixture(scope="class")
    def vllm_container(self, docker_client, vllm_image):
        """Creates and runs a container that is shared across tests in this class."""
        container = docker_client.containers.run(
            vllm_image.id,
            command="sleep 600",  # Keep container running for tests
            detach=True,
        )
        # Wait for the container to be in the 'running' state
        for _ in range(10):
            container.reload()
            if container.status == "running":
                break
            time.sleep(1)
        else:
            pytest.fail("Container did not start in time")

        yield container
        # Reliable cleanup
        try:
            container.remove(force=True)
        except docker.errors.APIError:
            pass  # Ignore errors during cleanup

    def test_image_builds_successfully(self, vllm_image):
        """Verify the Docker image was built and has an ID."""
        assert vllm_image is not None
        assert vllm_image.id is not None

    def test_image_has_required_components(self, vllm_container):
        """Verify the built image contains required paths and packages."""
        required_paths = ["/opt/code", "/opt/venv", "/opt/.home"]
        for path in required_paths:
            exit_code, _ = vllm_container.exec_run(f"test -e {path}")
            assert exit_code == 0, f"Required path not found in container: {path}"

        # Check for Python 3.12
        exit_code, _ = vllm_container.exec_run("python3.12 --version")
        assert exit_code == 0, "Python 3.12 not found in container"

        # Check that virtualenv was created
        exit_code, _ = vllm_container.exec_run("/opt/venv/bin/python --version")
        assert exit_code == 0, "Virtualenv Python not found in container"

    def test_container_runs_as_non_root(self, vllm_container):
        """Verify the container runs with the correct non-root user."""
        exit_code, output = vllm_container.exec_run("whoami")
        assert exit_code == 0
        # The user is 'datarobot' in the Dockerfile
        assert output.decode().strip() == "datarobot"

