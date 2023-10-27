#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from typing import Optional


def reduce_kwargs(input_dict, target_class):
    if not is_dataclass(target_class):
        return input_dict
    field_names = {field.name for field in fields(target_class)}
    return {k: v for k, v in input_dict.items() if k in field_names}


class AbstractSecret(ABC):
    @abstractmethod
    def is_partial_secret(self) -> bool:
        raise NotImplementedError()


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
        return bool(self.config_id) or bool(self.google_config_id)

    @classmethod
    def from_dict(cls, input_dict):
        reduced_dict = reduce_kwargs(input_dict, cls)
        gcp_key = "gcp_key"
        if isinstance(reduced_dict.get(gcp_key), dict):
            reduced_dict[gcp_key] = GCPKey.from_dict(reduced_dict[gcp_key])
        return cls(**reduced_dict)


def secrets_factory(input_dict: dict) -> AbstractSecret:
    return GCPSecret.from_dict(input_dict)
