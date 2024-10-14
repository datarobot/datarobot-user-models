from dataclasses import dataclass


class LazyLoadingEnvVars:
    @staticmethod
    def get_lazy_loading_data_key():
        return "MLOPS_LAZY_LOADING_DATA"

    @staticmethod
    def get_repository_credential_id_key_prefix():
        return "MLOPS_REPOSITORY_SECRET"


class BackendType:
    S3 = "s3"

    @staticmethod
    def all():
        return [BackendType.S3]
