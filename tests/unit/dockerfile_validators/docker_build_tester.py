"""
Docker build utilities for testing Dockerfiles.

This module provides utilities to test if Docker images can be built successfully
without requiring GPU hardware.
"""
import subprocess
import time
from typing import Optional, Tuple


class DockerBuildTester:
    """
    Utility for testing Docker image builds.

    This class provides methods to:
        - Build Docker images from Dockerfiles
        - Test basic commands in built images
        - Clean up test images after testing

    Attributes:
        dockerfile_dir: Directory containing the Dockerfile
        image_tag: Tag to use for the built image

    Example:
        >>> tester = DockerBuildTester('/path/to/dockerfile_dir', 'test-image:latest')
        >>> success, output, elapsed = tester.build_image()
        >>> if success:
        ...     cmd_success, cmd_output = tester.test_basic_command(['python', '--version'])
        >>> tester.cleanup()
    """

    def __init__(self, dockerfile_dir: str, image_tag: str):
        """
        Initialize the Docker build tester.

        Args:
            dockerfile_dir: Directory containing the Dockerfile
            image_tag: Tag to use for the built image (e.g., 'my-image:latest')
        """
        self.dockerfile_dir = dockerfile_dir
        self.image_tag = image_tag

    def build_image(self, timeout: Optional[int] = 600) -> Tuple[bool, str, float]:
        """
        Build Docker image from Dockerfile.

        Streams build output to console in real-time for long builds.
        Fails fast on Dockerfile syntax errors or build failures.

        Args:
            timeout: Maximum time to wait for build in seconds (default: 600)

        Returns:
            Tuple containing:
                - success (bool): True if build succeeded
                - output (str): Complete build output
                - elapsed_time (float): Build duration in seconds
        """
        cmd = ["docker", "build", "-t", self.image_tag, self.dockerfile_dir]
        start_time = time.time()
        output_lines = []
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in iter(process.stdout.readline, ""):
                print(line, end="")  # Stream output live
                output_lines.append(line)
            process.stdout.close()
            process.wait(timeout=timeout)
            elapsed_time = time.time() - start_time
            success = process.returncode == 0
            output = "".join(output_lines)
            return success, output, elapsed_time
        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            return False, f"Build timed out after {elapsed_time:.1f} seconds", elapsed_time

    def test_basic_command(self, command: list) -> Tuple[bool, str]:
        """
        Run a basic command in the built Docker image.

        Args:
            command: Command to run as list (e.g., ['python', '--version'])

        Returns:
            Tuple containing:
                - success (bool): True if command succeeded (exit code 0)
                - output (str): Combined stdout and stderr output
        """
        cmd = ["docker", "run", "--rm", "--entrypoint", command[0], self.image_tag] + command[1:]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        success = result.returncode == 0
        output = result.stdout + "\n" + result.stderr
        return success, output

    def cleanup(self) -> bool:
        """
        Remove the built Docker image.

        Returns:
            True if cleanup succeeded, False otherwise
        """
        cmd = ["docker", "rmi", self.image_tag]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
