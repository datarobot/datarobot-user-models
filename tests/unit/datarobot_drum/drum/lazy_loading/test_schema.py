import json

import pytest

from datarobot_drum.drum.lazy_loading.schema import LazyLoadingData
from datarobot_drum.drum.lazy_loading.schema import LazyLoadingFile
from datarobot_drum.drum.lazy_loading.schema import LazyLoadingRepository


class TestLazyLoadingSchema:
    @pytest.fixture
    def credential_ids(self):
        return ["670d0ea8c5549176219809ad", "670d0ea8c5549176219809ae"]

    @pytest.fixture
    def lazy_loading_files(self):
        return [
            LazyLoadingFile(
                remote_path="remote/path/large_file1.pkl",
                local_path="large_file1.pkl",
                repository_id="670d0ea8c55491762198099b",
            ),
            LazyLoadingFile(
                remote_path="remote/path/large_file2.pkl",
                local_path="large_file2.pkl",
                repository_id="670d0ea8c55491762198099c",
            ),
        ]

    @pytest.fixture
    def lazy_loading_repositories(self, credential_ids):
        return [
            LazyLoadingRepository(
                repository_id="670d0ea8c55491762198099b",
                bucket_name="my-bucket1",
                credential_id=credential_ids[0],
            ),
            LazyLoadingRepository(
                repository_id="670d0ea8c55491762198099c",
                bucket_name="my-bucket2",
                credential_id=credential_ids[1],
            ),
        ]

    @pytest.fixture
    def lazy_loading_data(self, lazy_loading_files, lazy_loading_repositories):
        return LazyLoadingData(files=lazy_loading_files, repositories=lazy_loading_repositories)

    def test_valid_json_success(self, lazy_loading_data):
        json_string = lazy_loading_data.json()
        loaded_data = LazyLoadingData.from_json_string(json_string)
        assert loaded_data == lazy_loading_data

    def test_extra_file_fields_are_ignored(self, lazy_loading_data):
        lazy_loading_data_dict = lazy_loading_data.dict()
        lazy_loading_data_dict["files"][0]["extra_filed"] = "extra"
        data_json = json.dumps(lazy_loading_data_dict)
        assert "extra_filed" in data_json

        loaded_data = LazyLoadingData.from_json_string(data_json)
        assert loaded_data == lazy_loading_data
        assert "extra_filed" not in loaded_data.json()

    def test_extra_repository_fields_are_ignored(self, lazy_loading_data):
        lazy_loading_data_dict = lazy_loading_data.dict()
        lazy_loading_data_dict["repositories"][0]["extra_filed"] = "extra"
        data_json = json.dumps(lazy_loading_data_dict)
        assert "extra_filed" in data_json

        loaded_data = LazyLoadingData.from_json_string(data_json)
        assert loaded_data == lazy_loading_data
        assert "extra_filed" not in loaded_data.json()

    def test_invalid_empty_endpoint_url(self, lazy_loading_files, lazy_loading_repositories):
        lazy_loading_repositories[0].endpoint_url = ""
        lazy_loading_data_json = LazyLoadingData(
            files=lazy_loading_files, repositories=lazy_loading_repositories
        ).json()
        with pytest.raises(ValueError, match="endpoint_url must not be an empty string"):
            LazyLoadingData.from_json_string(lazy_loading_data_json)

    def test_dependent_endpoint_url_and_verify_certificate_fields(
        self, lazy_loading_files, lazy_loading_repositories
    ):
        lazy_loading_repositories[0].endpoint_url = "https://minio.local:9000"
        lazy_loading_data_json = LazyLoadingData(
            files=lazy_loading_files, repositories=lazy_loading_repositories
        ).json()
        loaded_data = LazyLoadingData.from_json_string(lazy_loading_data_json)
        assert loaded_data.repositories[0].verify_certificate is True

    def test_missing_files_failure(self, lazy_loading_data):
        data_dict = lazy_loading_data.dict()
        del data_dict["files"]
        json_data = json.dumps(data_dict)
        with pytest.raises(ValueError, match="files"):
            LazyLoadingData.from_json_string(json_data)

    def test_missing_repositories_failure(self, lazy_loading_data):
        data_dict = lazy_loading_data.dict()
        del data_dict["repositories"]
        json_data = json.dumps(data_dict)
        with pytest.raises(ValueError, match="repositories"):
            LazyLoadingData.from_json_string(json_data)
