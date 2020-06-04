class DrumException(Exception):
    """Base drum exception"""

    pass


class DrumCommonException(DrumException):
    """Raised in case of common errors in drum"""

    pass


class DrumPerfTestTimeout(DrumException):
    """Raised when the perf-test case takes too long"""

    pass
