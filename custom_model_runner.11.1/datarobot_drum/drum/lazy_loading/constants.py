#
#  Copyright 2024 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
import json
from enum import Enum


class LazyLoadingEnvVars:
    @staticmethod
    def get_lazy_loading_data_key():
        return "MLOPS_LAZY_LOADING_DATA"

    @staticmethod
    def get_repository_credential_id_key_prefix():
        return "MLOPS_REPOSITORY_SECRET"


class BackendType(Enum):
    # WARNING: do not change the values of the enum members, because they are received from a
    # structured data in an environment variable.
    S3 = "s3"

    @staticmethod
    def all():
        return [BackendType.S3]


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)
