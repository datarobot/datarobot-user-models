#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import pytest

from datarobot_drum.custom_task_interfaces.user_secrets import GCPSecret, secrets_factory, GCPKey, \
    BasicSecret


class TestGCPSecret:
    def test_minimal_data(self):
        secret = {"credential_type": "gcp", "gcp_key": None}
        expected = GCPSecret(gcp_key=None)
        assert secrets_factory(secret) == expected

    def test_full_data(self):
        secret = {
            "credential_type": "gcp",
            "gcp_key": {"type": "abc"},
            "google_config_id": "abc",
            "config_id": "abc",
        }
        expected = GCPSecret(gcp_key=GCPKey(type="abc"), config_id="abc", google_config_id="abc",)
        assert secrets_factory(secret) == expected

    def test_is_partial_secret_false(self):
        assert not GCPSecret(gcp_key=None).is_partial_secret()

    @pytest.mark.parametrize("config_id, google_config_id", [
        ("abc", None), (None, "abc"), ("abc", "def")
    ])
    def test_is_partial_secret_true(self, config_id, google_config_id):
        gcp_secret = GCPSecret(gcp_key=None, google_config_id=google_config_id, config_id=config_id)
        assert gcp_secret.is_partial_secret()


    def test_extra_data(self):
        secret = {"credential_type": "gcp", "gcp_key": None, "ooops": "ac"}
        expected = GCPSecret(gcp_key=None)
        assert secrets_factory(secret) == expected

    def test_minimal_gcp_key(self):
        secret = {"credential_type": "gcp", "gcp_key": {"type": "stuff"}}
        expected = GCPSecret(gcp_key=GCPKey(type="stuff"))
        assert secrets_factory(secret) == expected

    def test_gcp_key_with_extra_fields(self):
        gcp_key = dict(type="abc", ooooops="abc")
        expected_key = GCPKey("abc",)
        secret = {"credential_type": "gcp", "gcp_key": gcp_key}
        expected = GCPSecret(gcp_key=expected_key)
        assert secrets_factory(secret) == expected

    def test_full_gcp_key(self):
        gcp_key = dict(
            type="abc",
            project_id="abc",
            private_key_id="abc",
            private_key="abc",
            client_email="abc",
            client_id="abc",
            auth_uri="abc",
            token_uri="abc",
            auth_provider_x509_cert_url="abc",
            client_x509_cert_url="abc",
        )
        expected_key = GCPKey("abc", "abc", "abc", "abc", "abc", "abc", "abc", "abc", "abc", "abc",)
        secret = {"credential_type": "gcp", "gcp_key": gcp_key}
        expected = GCPSecret(gcp_key=expected_key)
        assert secrets_factory(secret) == expected


class TestBasicSecret:
    def test_minimal_data(self):
        secret = {"credential_type": "basic", "username": "abc", "password": "def"}
        expected = BasicSecret(username="abc", password="def")
        assert secrets_factory(secret) == expected

    def test_full_data(self):
        secret = {"credential_type": "basic", "username": "abc", "password": "def", "snowflake_account_name": "ghi"}
        expected = BasicSecret(username="abc", password="def", snowflake_account_name="ghi")
        assert secrets_factory(secret) == expected

    def test_extra_data(self):
        secret = {"credential_type": "basic", "username": "abc", "password": "def", "ooops":"x"}
        expected = BasicSecret(username="abc", password="def")
        assert secrets_factory(secret) == expected

    def test_is_partial_secret(self):
        assert not BasicSecret(username="a", password="b").is_partial_secret()