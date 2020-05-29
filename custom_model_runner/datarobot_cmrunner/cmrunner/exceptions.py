class CMRunnerException(Exception):
    """Base cmrunner exception"""

    pass


class CMRunnerCommonException(CMRunnerException):
    """Raised in case of common errors in cmrunner"""

    pass


class CMRunnerPerfTestTimeout(CMRunnerException):
    """Raised when the perf-test case takes too long"""

    pass
