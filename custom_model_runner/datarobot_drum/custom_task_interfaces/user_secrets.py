#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import json
import logging
import os
import sys
from dataclasses import dataclass, fields, is_dataclass, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Optional, TypeVar, Type, Generic, Dict, Union, Set, List


def reduce_kwargs(input_dict: dict, target_class) -> dict:
    if not is_dataclass(target_class):
        return input_dict
    field_names = {field.name for field in fields(target_class)}
    return {k: v for k, v in input_dict.items() if k in field_names}


T = TypeVar("T")


@dataclass
class AbstractSecret(Generic[T]):
    def is_partial_secret(self) -> bool:
        """Some credentials from DataRobot contain admin-owned information.
        **That information will _not_ be available to the user in their secret**.
        This method tells you whether your credential has incomplete information because that information was
        stored separate from the credential itself."""
        config_keys = {"config_id", "oauth_config_id"}
        return any(getattr(self, key, None) for key in config_keys)

    @classmethod
    def from_dict(cls: Type[T], input_dict: dict) -> T:
        reduced = reduce_kwargs(input_dict, cls)
        return cls(**reduced)

    def to_dict(self):
        return asdict(self)


@dataclass()
class BasicSecret(AbstractSecret):
    username: str
    password: str
    snowflake_account_name: Optional[str] = None


@dataclass()
class OauthSecret(AbstractSecret):
    token: str
    refresh_token: str


@dataclass()
class S3Secret(AbstractSecret):
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    config_id: Optional[str] = None


@dataclass()
class AzureSecret(AbstractSecret):
    azure_connection_string: str


@dataclass()
class AzureServicePrincipalSecret(AbstractSecret):
    client_id: str
    client_secret: str
    azure_tenant_id: str


@dataclass()
class SnowflakeOauthUserAccountSecret(AbstractSecret):
    client_id: Optional[str]
    client_secret: Optional[str]
    snowflake_account_name: Optional[str]
    oauth_issuer_type: Optional[str] = None
    oauth_issuer_url: Optional[str] = None
    oauth_scopes: Optional[str] = None
    oauth_config_id: Optional[str] = None


@dataclass()
class SnowflakeKeyPairUserAccountSecret(AbstractSecret):
    username: Optional[str]
    private_key_str: Optional[str]
    passphrase: Optional[str] = None
    config_id: Optional[str] = None


@dataclass()
class AdlsGen2OauthSecret(AbstractSecret):
    client_id: str
    client_secret: str
    oauth_scopes: str


@dataclass()
class TableauAccessTokenSecret(AbstractSecret):
    token_name: str
    personal_access_token: str


@dataclass()
class DatabricksAccessTokenAccountSecret(AbstractSecret):
    databricks_access_token: str


@dataclass()
class ApiTokenSecret(AbstractSecret):
    api_token: str


@dataclass()
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


@dataclass()
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
    secrets = {k: secrets_factory(v) for k, v in all_secrets.items()}
    if secrets:
        sys.stdout = OutPutWrapper(secrets, sys.stdout)
        sys.stderr = OutPutWrapper(secrets, sys.stderr)
        ordered_values = get_ordered_sensitive_values(secrets)

        scrubbing_filter = ScrubberFilter(ordered_values)
        for logger in logging.root.manager.loggerDict.values():
            if hasattr(logger, "addFilter"):
                logger.addFilter(scrubbing_filter)
        logging.root.addFilter(scrubbing_filter)
        # TODO have a context manager that removes filter

    return secrets


def get_ordered_sensitive_values(secrets) -> List[str]:
    """This returns the set of all sensitive values, including recursing through
    sub-dictionaries so that they can be wiped from logs. Currently the only non-sensitive
    value is `credential_type`."""
    if not secrets:
        return []
    values_generator = (_get_all_values(secret.to_dict()) for secret in secrets.values())
    empty_set: Set[str] = set()
    all_values = empty_set.union(*values_generator)
    longest_first_to_replace_both_strings_and_sub_strings = sorted(
        all_values, key=lambda el: -len(el)
    )
    return longest_first_to_replace_both_strings_and_sub_strings


class OutPutWrapper:
    def __init__(self, secrets: Optional[Dict[str, AbstractSecret]], file_pointer):
        self.secret_values = get_ordered_sensitive_values(secrets)
        self.file_pointer = file_pointer

    def scrub_sensitive_data_from_string(self, input_str: str) -> str:
        for value in self.secret_values:
            input_str = input_str.replace(value, "*****")
        return input_str

    def write(self, intput_str):
        self.file_pointer.write(self.scrub_sensitive_data_from_string(intput_str))

    def flush(self):
        self.file_pointer.flush()


def scrub_sensitive_data(input_str, ordered_values):
    for value in ordered_values:
        input_str = input_str.replace(value, "****")
    return input_str


class ScrubberFilter(logging.Filter):
    def __init__(self, secrets):
        self.secrets = secrets
        super().__init__()

    def filter(self, record: logging.LogRecord):
        record.msg = self.transform(record.msg)
        if isinstance(record.args, dict):
            newargs = {k: self.transform(v) for k, v in record.args.items()}
        elif isinstance(record.args, tuple):
            newargs = tuple(self.transform(el) for el in record.args)
        else:
            newargs = self.transform(record.args)
        record.args = newargs
        return True

    def transform(self, element):
        if isinstance(element, AbstractSecret):
            element = str(element)
        
        if isinstance(element, str):
            return scrub_sensitive_data(element, self.secrets)
        return element

    def __eq__(self, other):
        if not isinstance(other, ScrubberFilter):
            return False
        return self.secrets == other.secrets

    def __hash__(self):
        return hash(" ".join(self.secrets))

# https://relaxdiego.com/2014/07/logging-in-python.html


def _get_all_values(actual_value: Union[str, dict]) -> Set[str]:
    if not actual_value:
        return set()
    if not isinstance(actual_value, dict):
        return {actual_value}
    sets_generator = (_get_all_values(el) for el in actual_value.values())
    empty_set: Set[str] = set()
    return empty_set.union(*sets_generator)


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
