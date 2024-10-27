from dataclasses import dataclass
from enum import Enum


class LazyLoadingEnvVars:
    @staticmethod
    def get_lazy_loading_data_key():
        return "MLOPS_LAZY_LOADING_DATA"

    @staticmethod
    def get_repository_credential_id_key_prefix():
        return "MLOPS_REPOSITORY_SECRET"


class BackendType(Enum):
    # WARNING: do not change the values of the enum members, because they are received from the
    # environment variables.
    S3 = "s3"

    @staticmethod
    def all():
        return [BackendType.S3]
