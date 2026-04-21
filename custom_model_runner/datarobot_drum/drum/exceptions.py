"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""


class DrumException(Exception):
    """Base drum exception"""


class DrumCommonException(DrumException):
    """Raised in case of common errors in drum"""


class DrumPerfTestTimeout(DrumException):
    """Raised when the perf-test case takes too long"""


class DrumPerfTestOOM(DrumException):
    """Raised when the container running drum during perf test is OOM"""


class DrumPredException(DrumException):
    """Raised when prediction consistency check fails"""


class DrumFormatSchemaException(DrumException):
    """Raised when supplied model_metadata cannot be parsed using strict yaml"""


class DrumSchemaValidationException(DrumException):
    """Raised when the supplied schema in model_metadata does not match actual input or output data."""


class DrumTransformException(DrumException):
    """Raised when there is an issue specific to transform tasks."""


class DrumSerializationError(DrumException):
    """Raised when there is an issue serializing or deserializing a custom task/model"""


class DrumRootComponentException(DrumException):
    """Raised when there is an issue specific to root components."""


class UnrecoverableError(DrumException):
    """A base exception for any error that is considered fatal and main runner should terminate immediately."""


class UnrecoverableConfigurationError(UnrecoverableError):
    """Raised when failure in parsing or validating configuration file."""


class ModelError(DrumException):
    """Raise this exception in your custom model to return a specific HTTP status code (400-499) to the user with custom message."""

    def __init__(self, message: str = "User error in custom model", status_code: int = 400):
        super().__init__(message)
        if not (400 <= status_code <= 499):
            raise ValueError(
                f"ModelError status_code must be between 400 and 499, got {status_code}"
            )
        self.status_code = status_code
