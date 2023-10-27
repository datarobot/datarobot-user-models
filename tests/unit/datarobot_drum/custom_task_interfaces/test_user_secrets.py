#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
from datarobot_drum.custom_task_interfaces.user_secrets import GCPSecret, secrets_factory, GCPKey


class TestGCPSecrets:
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
