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
from logging import Filter, LogRecord
from pathlib import Path
from typing import Optional, TypeVar, Type, Generic, Dict, Iterable, List, Set, Union, TextIO


def reduce_kwargs(input_dict, target_class):
    if not is_dataclass(target_class):
        return input_dict
    field_names = {field.name for field in fields(target_class)}
    return {k: v for k, v in input_dict.items() if k in field_names}


T = TypeVar("T")


@dataclass
class ScrubReprMixin:
    def __repr__(self):
        return f"{self.__class__.__name__}({self._get_args_string()})"

    def _get_args_string(self):
        return ", ".join(
            f"{field.name}={self._get_scrubbed_attribute(field)!r}" for field in fields(self)
        )

    def _get_scrubbed_attribute(self, field):
        raw_attribute = getattr(self, field.name)
        if isinstance(raw_attribute, str):
            return "*****"
        return raw_attribute


@dataclass(repr=False)
class AbstractSecret(ScrubReprMixin, Generic[T]):
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


@dataclass(repr=False)
class BasicSecret(AbstractSecret):
    username: str
    password: str
    snowflake_account_name: Optional[str] = None


@dataclass(repr=False)
class OauthSecret(AbstractSecret):
    token: str
    refresh_token: str


@dataclass(repr=False)
class S3Secret(AbstractSecret):
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    config_id: Optional[str] = None


@dataclass(repr=False)
class AzureSecret(AbstractSecret):
    azure_connection_string: str


@dataclass(repr=False)
class AzureServicePrincipalSecret(AbstractSecret):
    client_id: str
    client_secret: str
    azure_tenant_id: str


@dataclass(repr=False)
class SnowflakeOauthUserAccountSecret(AbstractSecret):
    client_id: Optional[str]
    client_secret: Optional[str]
    snowflake_account_name: Optional[str]
    oauth_issuer_type: Optional[str] = None
    oauth_issuer_url: Optional[str] = None
    oauth_scopes: Optional[str] = None
    oauth_config_id: Optional[str] = None


@dataclass(repr=False)
class SnowflakeKeyPairUserAccountSecret(AbstractSecret):
    username: Optional[str]
    private_key_str: Optional[str]
    passphrase: Optional[str] = None
    config_id: Optional[str] = None


@dataclass(repr=False)
class AdlsGen2OauthSecret(AbstractSecret):
    client_id: str
    client_secret: str
    oauth_scopes: str


@dataclass(repr=False)
class TableauAccessTokenSecret(AbstractSecret):
    token_name: str
    personal_access_token: str


@dataclass(repr=False)
class DatabricksAccessTokenAccountSecret(AbstractSecret):
    databricks_access_token: str


@dataclass(repr=False)
class ApiTokenSecret(AbstractSecret):
    api_token: str


@dataclass(repr=False)
class GCPKey(ScrubReprMixin):
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


@dataclass(repr=False)
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


def patch_outputs_to_scrub_secrets(secrets: Iterable[AbstractSecret]):
    if not secrets:
        return

    sys.stdout = TextStreamSecretsScrubber(secrets, sys.stdout)
    sys.stderr = TextStreamSecretsScrubber(secrets, sys.stderr)

    secrets_filter = SecretsScrubberFilter(secrets)
    for logger in _get_all_loggers():
        _safely_add_filter_to_logger(logger, secrets_filter)


def reset_outputs_to_allow_secrets():
    if isinstance(sys.stdout, TextStreamSecretsScrubber):
        sys.stdout = sys.stdout.stream
    if isinstance(sys.stderr, TextStreamSecretsScrubber):
        sys.stderr = sys.stderr.stream

    for logger in _get_all_loggers():
        _safely_remove_filters_from_logger(logger)


def _get_all_loggers():
    all_loggers = [logging.root.manager.root]
    all_loggers.extend(logging.root.manager.loggerDict.values())
    return all_loggers


def _safely_add_filter_to_logger(logger, secrets_filter):
    if hasattr(logger, "addFilter"):
        logger.addFilter(secrets_filter)


def _safely_remove_filters_from_logger(logger):
    if not hasattr(logger, "filters") or not hasattr(logger, "removeFilter"):
        return

    for logger_filter in logger.filters:
        if isinstance(logger_filter, SecretsScrubberFilter):
            logger.removeFilter(logger_filter)


class TextStreamSecretsScrubber:
    def __init__(self, secrets: Iterable[AbstractSecret], stream: TextIO):
        self.stream = stream
        self.sensitive_data = get_ordered_sensitive_values(secrets)

    def write(self, text):
        new_text = scrub_values_from_string(self.sensitive_data, text)
        self.stream.write(new_text)

    def writelines(self, lines):
        new_lines = [scrub_values_from_string(self.sensitive_data, text) for text in lines]
        self.stream.writelines(new_lines)

    def __getattr__(self, item):
        return object.__getattribute__(self.stream, item)

    def __eq__(self, other):
        if not isinstance(other, TextStreamSecretsScrubber):
            return False
        return (self.stream, self.sensitive_data) == (other.stream, other.sensitive_data)


class SecretsScrubberFilter(Filter):
    def __init__(self, secrets: Iterable[AbstractSecret]):
        super().__init__()
        self.sensitive_data = get_ordered_sensitive_values(secrets)

    def filter(self, record: LogRecord):
        record.msg = self.transform(record.msg)
        if isinstance(record.args, dict):
            new_args = {k: self.transform(v) for k, v in record.args.items()}
        elif isinstance(record.args, tuple):
            new_args = tuple(self.transform(el) for el in record.args)
        else:
            new_args = self.transform(record.args)
        record.args = new_args
        return True

    def transform(self, element):
        if isinstance(element, str):
            return scrub_values_from_string(self.sensitive_data, element)
        else:
            return element

    def __eq__(self, other):
        if not isinstance(other, SecretsScrubberFilter):
            return False
        return self.sensitive_data == other.sensitive_data


def get_ordered_sensitive_values(secrets: Iterable[AbstractSecret]) -> List[str]:
    """This returns the list of all sensitive values, including recursing through
    sub-dictionaries so that they can be wiped from logs."""
    if not secrets:
        return []
    values_generator = (_get_all_values(asdict(secret)) for secret in secrets)
    empty_set: Set[str] = set()
    all_values = empty_set.union(*values_generator)
    longest_first_to_replace_both_strings_and_sub_strings = sorted(
        all_values, key=lambda el: (-len(el), el)
    )
    return longest_first_to_replace_both_strings_and_sub_strings


def _get_all_values(actual_value: Union[str, dict]) -> Set[str]:
    if not actual_value:
        return set()
    if not isinstance(actual_value, dict):
        return {actual_value}
    sets_generator = (_get_all_values(el) for el in actual_value.values())
    empty_set: Set[str] = set()
    return empty_set.union(*sets_generator)


def scrub_values_from_string(sensitive_values: List[str], input_str: str) -> str:
    for value in sensitive_values:
        input_str = input_str.replace(value, "*****")
    return input_str
