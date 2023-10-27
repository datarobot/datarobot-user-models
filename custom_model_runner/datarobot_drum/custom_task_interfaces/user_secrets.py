#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, auto
from typing import Optional, TypeVar, Type, Generic


def reduce_kwargs(input_dict, target_class):
    if not is_dataclass(target_class):
        return input_dict
    field_names = {field.name for field in fields(target_class)}
    return {k: v for k, v in input_dict.items() if k in field_names}


T = TypeVar("T")


class AbstractSecret(Generic[T], ABC):
    @abstractmethod
    def is_partial_secret(self) -> bool:
        raise NotImplementedError()

    @classmethod
    def from_dict(cls: Type[T], input_dict) -> T:
        reduced = reduce_kwargs(input_dict, cls)
        return cls(**reduced)


class SecretWithoutAnyConfig(AbstractSecret):
    def is_partial_secret(self) -> bool:
        return False


@dataclass(frozen=True)
class BasicSecret(SecretWithoutAnyConfig):
    username: str
    password: str
    snowflake_account_name: Optional[str] = None


@dataclass(frozen=True)
class OauthSecret(SecretWithoutAnyConfig):
    token: str
    refresh_token: str


@dataclass(frozen=True)
class S3Secret(AbstractSecret):
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    config_id: Optional[str] = None

    def is_partial_secret(self) -> bool:
        return bool(self.config_id)


@dataclass(frozen=True)
class AzureSecret(SecretWithoutAnyConfig):
    azure_connection_string: str


@dataclass(frozen=True)
class AzureServicePrincipalSecret(SecretWithoutAnyConfig):
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

    def is_partial_secret(self) -> bool:
        return bool(self.oauth_config_id)


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
    google_config_id: Optional[str] = None
    config_id: Optional[str] = None

    def is_partial_secret(self) -> bool:
        return bool(self.config_id or self.google_config_id)

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

    @classmethod
    def from_string(cls, input_string: str) -> "SecretType":
        return getattr(cls, input_string.upper())

    def get_secret_class(self) -> Type[AbstractSecret]:
        mapping = {
            self.BASIC: BasicSecret,
            self.OAUTH: OauthSecret,
            self.GCP: GCPSecret,
            self.S3: S3Secret,
            self.AZURE: AzureSecret,
            self.AZURE_SERVICE_PRINCIPAL: AzureServicePrincipalSecret,
            self.SNOWFLAKE_OAUTH_USER_ACCOUNT: SnowflakeOauthUserAccountSecret,
        }
        return mapping[self]


def secrets_factory(input_dict: dict) -> AbstractSecret:
    secret_type = SecretType.from_string(input_dict["credential_type"])
    return secret_type.get_secret_class().from_dict(input_dict)
