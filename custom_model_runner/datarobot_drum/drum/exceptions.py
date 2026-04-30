"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

HTTP_BAD_REQUEST = 400


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


class BaseCustomUserError(Exception):
    """Base class for errors that originate in user-provided custom model code."""


class CustomHTTPError(BaseCustomUserError):
    """Raise this exception in your custom model to return a specific HTTP status code with custom message."""

    def __init__(
        self,
        message: str = "User's HTTP error in custom model",
        status_code: int = HTTP_BAD_REQUEST,
    ):
        super().__init__(message)
        self.status_code = status_code
