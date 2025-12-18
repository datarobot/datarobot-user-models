#
#  Copyright 2024 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import conlist
from pydantic import constr
from pydantic import model_validator

from datarobot_drum.drum.lazy_loading.constants import BackendType


def to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class LazyLoadingFile(BaseModel):
    remote_path: constr(min_length=1)
    local_path: constr(min_length=1)
    repository_id: constr(min_length=1)

    class Config:
        extra = "ignore"
        alias_generator = to_camel
        populate_by_name = True


class LazyLoadingRepository(BaseModel):
    repository_id: constr(min_length=1)
    bucket_name: constr(min_length=1)
    credential_id: constr(min_length=1)
    endpoint_url: Optional[str] = None
    verify_certificate: Optional[bool] = None

    class Config:
        extra = "ignore"
        alias_generator = to_camel
        populate_by_name = True

    @model_validator(mode="before")
    @classmethod
    def validate_model_attributes(cls, data):
        endpoint_url = data.get("endpoint_url")
        if endpoint_url is None:
            return data
        if endpoint_url == "":
            raise ValueError("endpoint_url must not be an empty string")
        if data.get("verify_certificate") is None:
            data["verify_certificate"] = True
        return data


class LazyLoadingData(BaseModel):
    files: conlist(LazyLoadingFile, min_length=1)
    repositories: conlist(LazyLoadingRepository, min_length=1)

    class Config:
        extra = "ignore"
        alias_generator = to_camel
        populate_by_name = True

    @classmethod
    def from_json_string(cls, json_string: str):
        return cls.model_validate_json(json_string)


class S3Credentials(BaseModel):
    credential_type: BackendType
    aws_access_key_id: constr(min_length=1)
    aws_secret_access_key: constr(min_length=1)
    aws_session_token: Optional[constr(min_length=1)] = None

    class Config:
        # Future proofing in case we want to add a profile field
        extra = "ignore"
        alias_generator = to_camel
        populate_by_name = True


class LazyLoadingCommandLineFileCredentialsContent(S3Credentials):
    id: constr(min_length=1)

    class Config:
        # Future proofing in case we want to add a profile field
        extra = "ignore"
        alias_generator = to_camel
        populate_by_name = True


class LazyLoadingCommandLineFileContent(LazyLoadingData):
    credentials: List[LazyLoadingCommandLineFileCredentialsContent]

    class Config:
        # Future proofing in case we want to add a profile field
        extra = "ignore"
        alias_generator = to_camel
        populate_by_name = True

    @classmethod
    def from_json_string(cls, json_string: str):
        return cls.model_validate_json(json_string)
