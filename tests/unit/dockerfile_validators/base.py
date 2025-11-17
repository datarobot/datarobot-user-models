"""
Base classes and utilities for Dockerfile validation.

This module provides reusable components for validating Dockerfile configurations
across different dropin environments without requiring GPU hardware.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple


class EnvVarValidator:
    """Base validators for common environment variable types."""

    @staticmethod
    def is_boolean_flag(value: str) -> bool:
        """
        Validate boolean flags for Dockerfile.

        At runtime, DRUM uses trafaret.ToBool() which accepts:
        - True values: '1', 'true', 'yes', 'y', 'on' (case-insensitive)
        - False values: '0', 'false', 'no', 'n', 'off' (case-insensitive)

        This validator matches the actual runtime behavior.

        Args:
            value: Environment variable value to validate

        Returns:
            True if value is accepted by trafaret.ToBool(), False otherwise

        Note:
            Matches DRUM's runtime behavior using trafaret.ToBool()
        """
        value_lower = value.lower()
        true_values = {"1", "true", "yes", "y", "on"}
        false_values = {"0", "false", "no", "n", "off"}
        return value_lower in (true_values | false_values)

    @staticmethod
    def is_valid_path(value: str) -> bool:
        """
        Validate Unix-style path format for Dockerfile ENV variables.

        Note: This validates the path format only, not existence, since these are
        container paths that don't exist on the host filesystem during validation.

        Args:
            value: Path value to validate

        Returns:
            True if value is a valid path format (absolute path or variable reference)
        """
        if not value or value.isspace():
            return False
        # Allow environment variable references
        if value.startswith("$"):
            return True
        # Require absolute paths (container paths)
        if not value.startswith("/"):
            return False
        # Reject paths with trailing spaces (common typo)
        if value.endswith(" "):
            return False
        # Reject paths with invalid characters
        if any(char in value for char in ["\n", "\r", "\t"]):
            return False
        return True

    @staticmethod
    def is_valid_address(value: str) -> bool:
        """
        Validate network address in IP:PORT or localhost:PORT format.

        Args:
            value: Address value to validate

        Returns:
            True if value matches expected network address format (IP:PORT or localhost:PORT)
        """
        pattern = r"^(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|localhost):\d{1,5}$"
        return bool(re.match(pattern, value))

    @staticmethod
    def is_valid_path_with_vars(value: str) -> bool:
        """
        Validate PATH-style environment variables with colon-separated paths.

        Validates format like: /usr/bin:/usr/local/bin:$HOME/bin

        Args:
            value: PATH value to validate

        Returns:
            True if value is properly formatted PATH-style variable
        """
        if not value or value.isspace():
            return False
        # Reject trailing/leading spaces (common typos)
        if value != value.strip():
            return False
        # Reject invalid characters
        if any(char in value for char in ["\n", "\r", "\t"]):
            return False
        # Split by colon and validate each path component
        paths = value.split(":")
        for path in paths:
            if not path:  # Empty component like /usr/bin::local
                return False
            # Each component should be absolute path or variable reference
            if not (path.startswith("/") or path.startswith("$")):
                return False
        return True


class DockerfileParser:
    """
    Parser for extracting environment variables from Dockerfiles.

    Handles multiple ENV declaration formats:
        - ENV KEY=VALUE
        - ENV KEY1=VALUE1 KEY2=VALUE2
        - ENV KEY VALUE (deprecated but valid)
    """

    @staticmethod
    def parse_env_vars(dockerfile_path: str) -> Dict[str, str]:
        """
        Parse all ENV declarations from a Dockerfile.

        Args:
            dockerfile_path: Absolute path to the Dockerfile

        Returns:
            Dictionary mapping environment variable names to their values
        """
        env_vars = {}
        with open(dockerfile_path, "r") as f:
            content = f.read()
        pattern = r"^\s*ENV\s+(.+)$"
        for line in content.split("\n"):
            match = re.match(pattern, line)
            if match:
                env_declaration = match.group(1).strip()
                if "=" in env_declaration:
                    parts = env_declaration.split()
                    for part in parts:
                        if "=" in part:
                            key, value = part.split("=", 1)
                            env_vars[key.strip()] = value.strip()
                else:
                    parts = env_declaration.split(None, 1)
                    if len(parts) == 2:
                        env_vars[parts[0].strip()] = parts[1].strip()
        return env_vars


class BaseDockerfileValidator(ABC):
    """
    Abstract base class for validating Dockerfile environment variables.

    This class provides a framework for validating ENV declarations in Dockerfiles
    without requiring Docker or GPU hardware. Subclasses must implement:
        - env_validation_rules: Rules for each environment variable
        - typo_patterns: Common typo patterns to detect

    Attributes:
        dockerfile_path: Path to the Dockerfile being validated
        parser: DockerfileParser instance for extracting ENV variables
    """

    def __init__(self, dockerfile_path: str):
        """
        Initialize the validator.

        Args:
            dockerfile_path: Absolute path to the Dockerfile to validate
        """
        self.dockerfile_path = dockerfile_path
        self.parser = DockerfileParser()

    @property
    @abstractmethod
    def env_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Define validation rules for environment variables.

        Returns:
            Dictionary mapping variable names to validation rules.
            Each rule must contain:
                - type (str): Variable type (e.g., 'boolean', 'path')
                - validator (callable): Function to validate the value
                - description (str): Human-readable description
                - required (bool, optional): Whether variable is required
                - valid_values (list, optional): List of acceptable values

        Example:
            {
                'WITH_ERROR_SERVER': {
                    'type': 'boolean',
                    'validator': EnvVarValidator.is_boolean_flag,
                    'valid_values': ['0', '1', 'true', 'false', ...],
                    'description': 'Error server toggle flag',
                    'required': True
                }
            }
        """
        pass

    @property
    @abstractmethod
    def typo_patterns(self) -> List[Tuple[str, str]]:
        """
        Define regex patterns for common typos.

        Returns:
            List of (pattern, description) tuples.
            Each tuple contains:
                - pattern (str): Regex pattern to match against values
                - description (str): Human-readable description of the issue

        Example:
            [
                (r'\\s+$', 'Trailing whitespace'),
                (r'0s$', "Possible typo: '0s' instead of '0'")
            ]
        """
        pass

    def validate_env_vars(self) -> List[str]:
        """
        Validate all environment variables against defined rules.

        Checks each ENV variable for:
            - Correct type (boolean, path, address, etc.)
            - Valid values (if specified in rules)
            - Required variables are present

        Returns:
            List of error messages (empty if all variables are valid)
        """
        env_vars = self.parser.parse_env_vars(self.dockerfile_path)
        validation_errors = []
        for var_name, var_value in sorted(env_vars.items()):
            if var_name not in self.env_validation_rules:
                continue
            rule = self.env_validation_rules[var_name]
            validator = rule["validator"]
            if not validator(var_value):
                error_msg = (
                    f"\n  Variable: {var_name}\n"
                    f"  Value: '{var_value}'\n"
                    f"  Type: {rule['type']}\n"
                    f"  Description: {rule['description']}"
                )
                if "valid_values" in rule:
                    error_msg += f"\n  Valid values: {rule['valid_values']}"
                validation_errors.append(error_msg)
        defined_vars = set(env_vars.keys())
        required_vars = {
            var for var, rule in self.env_validation_rules.items() if rule.get("required", False)
        }
        missing_vars = required_vars - defined_vars
        if missing_vars:
            validation_errors.append(
                f"\nMissing required environment variables: {', '.join(sorted(missing_vars))}"
            )
        return validation_errors

    def check_typos(self) -> List[str]:
        """
        Check for common typos in environment variable values.

        Uses regex patterns defined in typo_patterns to detect issues like:
            - Trailing/leading whitespace
            - Common typos (e.g., '0s' instead of '0')
            - Unexpected suffixes or prefixes

        Returns:
            List of error messages (empty if no typos found)
        """
        env_vars = self.parser.parse_env_vars(self.dockerfile_path)
        typo_errors = []
        for var_name, var_value in sorted(env_vars.items()):
            if var_name not in self.env_validation_rules:
                continue
            for pattern, description in self.typo_patterns:
                if re.search(pattern, var_value):
                    typo_errors.append(f"\n  {var_name}='{var_value}'\n" f"  Issue: {description}")
        return typo_errors
