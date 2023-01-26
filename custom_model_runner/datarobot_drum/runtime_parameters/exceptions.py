"""
Copyright 2023 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""


class RuntimeParameterException(Exception):
    pass


class InvalidJsonException(RuntimeParameterException):
    pass


class InvalidRuntimeParam(RuntimeParameterException):
    pass


class InvalidInputFilePath(RuntimeParameterException):
    pass


class InvalidEmptyYamlContent(RuntimeParameterException):
    pass


class InvalidYamlContent(RuntimeParameterException):
    pass


class ErrorLoadingRuntimeParameter(RuntimeParameterException):
    pass
