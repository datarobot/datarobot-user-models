#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import json
import os
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, TypeVar, Type, Generic, Dict


def reduce_kwargs(input_dict, target_class):
    if not is_dataclass(target_class):
        return input_dict
    field_names = {field.name for field in fields(target_class)}
    return {k: v for k, v in input_dict.items() if k in field_names}


T = TypeVar("T")


class AbstractSecret(Generic[T]):
    def is_partial_secret(self) -> bool:
        """Some credentials from DataRobot contain admin-owned information.
        **That information will _not_ be available to the user in their secret**.
        This method tells you whether your credential has incomplete information because that information was
        stored separate from the credential itself."""
        config_keys = {"config_id", "oauth_config_id"}
        return any(getattr(self, key, None) for key in config_keys)

    @classmethod
    def from_dict(cls: Type[T], input_dict) -> T:
        reduced = reduce_kwargs(input_dict, cls)
        return cls(**reduced)


@dataclass(frozen=True)
class BasicSecret(AbstractSecret):
    username: str
    password: str
    snowflake_account_name: Optional[str] = None


@dataclass(frozen=True)
class OauthSecret(AbstractSecret):
    token: str
    refresh_token: str


@dataclass(frozen=True)
class S3Secret(AbstractSecret):
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    config_id: Optional[str] = None


@dataclass(frozen=True)
class AzureSecret(AbstractSecret):
    azure_connection_string: str


@dataclass(frozen=True)
class AzureServicePrincipalSecret(AbstractSecret):
    client_id: str
    client_secret: str
    azure_tenant_id: str


@dataclass(frozen=True)
class SnowflakeOauthUserAccountSecret(AbstractSecret):
    client_id: Optional[str]
    client_secret: Optional[str]
    snowflake_account_name: Optional[str]
    oauth_issuer_type: Optional[str] = None
    oauth_issuer_url: Optional[str] = None
    oauth_scopes: Optional[str] = None
    oauth_config_id: Optional[str] = None


@dataclass(frozen=True)
class SnowflakeKeyPairUserAccountSecret(AbstractSecret):
    username: Optional[str]
    private_key_str: Optional[str]
    passphrase: Optional[str] = None
    config_id: Optional[str] = None


@dataclass(frozen=True)
class AdlsGen2OauthSecret(AbstractSecret):
    client_id: str
    client_secret: str
    oauth_scopes: str


@dataclass(frozen=True)
class TableauAccessTokenSecret(AbstractSecret):
    token_name: str
    personal_access_token: str


@dataclass(frozen=True)
class DatabricksAccessTokenAccountSecret(AbstractSecret):
    databricks_access_token: str


@dataclass(frozen=True)
class ApiTokenSecret(AbstractSecret):
    api_token: str


@dataclass(frozen=True)
class GCPKey:
    type: str
    project_id: Optional[str] = None
    private_key_id: Optional[str] = None
    private_key: Optional[str] = None
    client_email: Optional[str] = None
    client_id: Optional[str] = None
    auth_uri: Optional[str] = None
    token_uri: Optional[str] = None
    auth_provider_x509_cert_url: Optional[str] = None
    client_x509_cert_url: Optional[str] = None

    @classmethod
    def from_dict(cls, input_dict):
        return cls(**reduce_kwargs(input_dict, cls))


@dataclass(frozen=True)
class GCPSecret(AbstractSecret):
    gcp_key: Optional[GCPKey] = None
    config_id: Optional[str] = None

    @classmethod
    def from_dict(cls, input_dict):
        reduced_dict = reduce_kwargs(input_dict, cls)
        gcp_key = "gcp_key"
        if isinstance(reduced_dict.get(gcp_key), dict):
            reduced_dict[gcp_key] = GCPKey.from_dict(reduced_dict[gcp_key])
        return cls(**reduced_dict)


class SecretType(Enum):
    BASIC = auto()
    OAUTH = auto()
    GCP = auto()
    S3 = auto()
    AZURE = auto()
    AZURE_SERVICE_PRINCIPAL = auto()
    SNOWFLAKE_OAUTH_USER_ACCOUNT = auto()
    SNOWFLAKE_KEY_PAIR_USER_ACCOUNT = auto()
    ADLS_GEN2_OAUTH = auto()
    TABLEAU_ACCESS_TOKEN = auto()
    DATABRICKS_ACCESS_TOKEN_ACCOUNT = auto()
    API_TOKEN = auto()

    @classmethod
    def from_string(cls, input_string: str) -> "SecretType":
        try:
            return getattr(cls, input_string.upper())
        except AttributeError:
            raise UnsupportedSecretError(f"Unsupported secret type: {input_string!r}")

    def get_secret_class(self) -> Type[AbstractSecret]:
        mapping = {
            self.BASIC: BasicSecret,
            self.OAUTH: OauthSecret,
            self.GCP: GCPSecret,
            self.S3: S3Secret,
            self.AZURE: AzureSecret,
            self.AZURE_SERVICE_PRINCIPAL: AzureServicePrincipalSecret,
            self.SNOWFLAKE_OAUTH_USER_ACCOUNT: SnowflakeOauthUserAccountSecret,
            self.SNOWFLAKE_KEY_PAIR_USER_ACCOUNT: SnowflakeKeyPairUserAccountSecret,
            self.ADLS_GEN2_OAUTH: AdlsGen2OauthSecret,
            self.TABLEAU_ACCESS_TOKEN: TableauAccessTokenSecret,
            self.DATABRICKS_ACCESS_TOKEN_ACCOUNT: DatabricksAccessTokenAccountSecret,
            self.API_TOKEN: ApiTokenSecret,
        }
        return mapping[self]


class UnsupportedSecretError(Exception):
    pass


def secrets_factory(input_dict: dict) -> AbstractSecret:
    """
    Casts all secrets to AbstractSecret

    Raises
    ------
    UnsupportedSecretError
    """
    secret_type = SecretType.from_string(input_dict["credential_type"])
    return secret_type.get_secret_class().from_dict(input_dict)


def load_secrets(
    mount_path: Optional[str], env_var_prefix: Optional[str]
) -> Dict[str, AbstractSecret]:
    all_secrets = {}
    env_secrets = _get_environment_secrets(env_var_prefix)
    mounted_secrets = _get_mounted_secrets(mount_path)
    all_secrets.update(env_secrets)
    all_secrets.update(mounted_secrets)
    return {k: secrets_factory(v) for k, v in all_secrets.items()}


def _get_environment_secrets(env_var_prefix):
    if not env_var_prefix:
        return {}

    full_prefix = f"{env_var_prefix}_"
    actual_secrets = [(k, v) for k, v in os.environ.items() if k.startswith(full_prefix)]

    return {key.replace(full_prefix, ""): json.loads(value) for key, value in actual_secrets}


def _get_mounted_secrets(mount_path: Optional[str]):
    if mount_path is None:
        return {}

    secret_files = [file_path for file_path in Path(mount_path).glob("*") if file_path.is_file()]

    def get_dict(file_path: Path):
        with file_path.open() as fp:
            return json.load(fp)

    return {file_path.name: get_dict(file_path) for file_path in secret_files}
