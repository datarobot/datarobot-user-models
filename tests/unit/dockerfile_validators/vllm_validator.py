from tests.unit.dockerfile_validators.base import BaseDockerfileValidator, EnvVarValidator
from typing import Dict, Any, List, Tuple


class VllmDockerfileValidator(BaseDockerfileValidator):
    """
    Validator for vLLM Dockerfile environment variables.
    """

    @property
    def env_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        BOOLEAN_VALUES = ["0", "1", "true", "false", "yes", "no", "y", "n", "on", "off"]
        return {
            "WITH_ERROR_SERVER": {
                "type": "boolean",
                "validator": EnvVarValidator.is_boolean_flag,
                "valid_values": BOOLEAN_VALUES,
                "description": "Error server toggle flag",
                "required": True,
                "source": "ArgumentOptionsEnvVars.BOOL_VARS",
                "note": "Accepts any case (True, TRUE, true, etc.)",
            },
            "DO_NOT_TRACK": {
                "type": "boolean",
                "validator": EnvVarValidator.is_boolean_flag,
                "valid_values": BOOLEAN_VALUES,
                "description": "Telemetry tracking flag (vLLM/HuggingFace)",
                "required": True,
                "source": "Dockerfile comment and vLLM convention",
                "note": "Accepts any case (True, TRUE, true, etc.)",
            },
            "PIP_NO_CACHE_DIR": {
                "type": "boolean",
                "validator": EnvVarValidator.is_boolean_flag,
                "valid_values": BOOLEAN_VALUES,
                "description": "Pip cache control flag",
                "required": True,
                "source": "Standard pip environment variable",
                "note": "Accepts any case (True, TRUE, true, etc.)",
            },
            "PYTHONUNBUFFERED": {
                "type": "boolean",
                "validator": EnvVarValidator.is_boolean_flag,
                "valid_values": BOOLEAN_VALUES,
                "description": "Python output buffering flag",
                "required": True,
                "source": "Standard Python environment variable",
                "note": "Accepts any case (True, TRUE, true, etc.)",
            },
            "CODE_DIR": {
                "type": "path",
                "validator": EnvVarValidator.is_valid_path,
                "description": "Code directory path",
                "required": True,
                "source": "ArgumentOptionsEnvVars.VALUE_VARS",
            },
            "DATAROBOT_VENV_PATH": {
                "type": "path",
                "validator": EnvVarValidator.is_valid_path,
                "description": "Virtual environment path",
                "required": True,
                "source": "Custom DataRobot convention",
            },
            "HOME": {
                "type": "path",
                "validator": EnvVarValidator.is_valid_path,
                "description": "Home directory path",
                "required": True,
                "source": "Standard Unix environment variable",
            },
            "VIRTUAL_ENV": {
                "type": "path",
                "validator": EnvVarValidator.is_valid_path,
                "description": "Virtual environment path variable",
                "required": True,
                "source": 'Used in vllm_predictor.py: os.environ.get("VIRTUAL_ENV")',
            },
            "ADDRESS": {
                "type": "address",
                "validator": EnvVarValidator.is_valid_address,
                "description": "Server bind address and port",
                "required": True,
                "source": "ArgumentOptionsEnvVars.VALUE_VARS",
            },
            "PATH": {
                "type": "path_with_vars",
                "validator": EnvVarValidator.is_valid_path_with_vars,
                "description": "System PATH variable",
                "required": False,
                "source": "Standard Unix environment variable",
            },
        }

    @property
    def typo_patterns(self) -> List[Tuple[str, str]]:
        return [
            (r"\s+$", "Trailing whitespace"),
            (r"^\s+", "Leading whitespace"),
            (r"0s$", "Possible typo: '0s' instead of '0'"),
        ]
