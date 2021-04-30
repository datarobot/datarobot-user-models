class DrumException(Exception):
    """Base drum exception"""

    pass


class DrumCommonException(DrumException):
    """Raised in case of common errors in drum"""

    pass


class DrumPerfTestTimeout(DrumException):
    """Raised when the perf-test case takes too long"""

    pass


class DrumPerfTestOOM(DrumException):
    """ Raised when the container running drum during perf test is OOM """

    pass


class DrumPredException(DrumException):
    """ Raised when prediction consistency check fails"""

    pass


class DrumSchemaValidationException(DrumException):
    """ Raised when the supplied schema in model_metadata does not match actual input or output data."""
