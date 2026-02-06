#
#  Copyright 2024 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
from typing import Annotated
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from datarobot_drum.drum.lazy_loading.constants import BackendType

# Type aliases for Pydantic v2
NonEmptyStr = Annotated[str, Field(min_length=1)]


def to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class LazyLoadingFile(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    remote_path: NonEmptyStr
    local_path: NonEmptyStr
    repository_id: NonEmptyStr


class LazyLoadingRepository(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    repository_id: NonEmptyStr
    bucket_name: NonEmptyStr
    credential_id: NonEmptyStr
    endpoint_url: Optional[str] = None
    verify_certificate: Optional[bool] = None

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
    model_config = ConfigDict(
        extra="ignore",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    files: Annotated[List[LazyLoadingFile], Field(min_length=1)]
    repositories: Annotated[List[LazyLoadingRepository], Field(min_length=1)]

    @classmethod
    def from_json_string(cls, json_string: str):
        return cls.model_validate_json(json_string)


class S3Credentials(BaseModel):
    model_config = ConfigDict(
        # Future proofing in case we want to add a profile field
        extra="ignore",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    credential_type: BackendType
    aws_access_key_id: NonEmptyStr
    aws_secret_access_key: NonEmptyStr
    aws_session_token: Optional[NonEmptyStr] = None


class LazyLoadingCommandLineFileCredentialsContent(S3Credentials):
    model_config = ConfigDict(
        # Future proofing in case we want to add a profile field
        extra="ignore",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    id: NonEmptyStr


class LazyLoadingCommandLineFileContent(LazyLoadingData):
    model_config = ConfigDict(
        # Future proofing in case we want to add a profile field
        extra="ignore",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    credentials: List[LazyLoadingCommandLineFileCredentialsContent]

    @classmethod
    def from_json_string(cls, json_string: str):
        return cls.model_validate_json(json_string)
