from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import conlist
from pydantic import constr
from pydantic import model_validator

from datarobot_drum.drum.lazy_loading.constants import BackendType


class LazyLoadingFile(BaseModel):
    remote_path: constr(min_length=1)
    local_path: constr(min_length=1)
    repository_id: constr(min_length=1)

    class Config:
        extra = "ignore"


class LazyLoadingRepository(BaseModel):
    repository_id: constr(min_length=1)
    bucket_name: constr(min_length=1)
    credential_id: constr(min_length=1)
    endpoint_url: Optional[str] = None
    verify_certificate: Optional[bool] = None

    class Config:
        extra = "ignore"

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

    @classmethod
    def from_json_string(cls, json_string: str):
        return cls.parse_raw(json_string)


class S3Credentials(BaseModel):
    credential_type: BackendType
    aws_access_key_id: constr(min_length=1)
    aws_secret_access_key: constr(min_length=1)
    aws_session_token: Optional[constr(min_length=1)] = None

    class Config:
        # Future proofing in case we want to add a profile field
        extra = "ignore"
