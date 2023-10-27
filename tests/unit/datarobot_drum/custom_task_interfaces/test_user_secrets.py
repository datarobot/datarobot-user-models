#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#

from datarobot_drum.custom_task_interfaces.user_secrets import (
    GCPSecret,
    secrets_factory,
    GCPKey,
    BasicSecret,
    OauthSecret,
    S3Secret,
    AzureSecret,
    AzureServicePrincipalSecret,
    SnowflakeOauthUserAccountSecret,
    SnowflakeKeyPairUserAccountSecret,
    AdlsGen2OauthSecret,
    TableauAccessTokenSecret,
)


class TestGCPSecret:
    def test_minimal_data(self):
        secret = {"credential_type": "gcp", "gcp_key": None}
        expected = GCPSecret(gcp_key=None)
        assert secrets_factory(secret) == expected

    def test_full_data(self):
        secret = {
            "credential_type": "gcp",
            "gcp_key": {"type": "abc"},
            "config_id": "abc",
        }
        expected = GCPSecret(gcp_key=GCPKey(type="abc"), config_id="abc")
        assert secrets_factory(secret) == expected

    def test_is_partial_secret_false(self):
        assert not GCPSecret(gcp_key=None, config_id=None).is_partial_secret()

    def test_is_partial_secret_true(self):
        gcp_secret = GCPSecret(gcp_key=None, config_id="abc")
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
        secret = {
            "credential_type": "basic",
            "username": "abc",
            "password": "def",
            "snowflake_account_name": "ghi",
        }
        expected = BasicSecret(username="abc", password="def", snowflake_account_name="ghi")
        assert secrets_factory(secret) == expected

    def test_extra_data(self):
        secret = {"credential_type": "basic", "username": "abc", "password": "def", "ooops": "x"}
        expected = BasicSecret(username="abc", password="def")
        assert secrets_factory(secret) == expected

    def test_is_partial_secret(self):
        assert not BasicSecret(username="a", password="b").is_partial_secret()


class TestOauthSecret:
    def test_minimal_data(self):
        secret = {"credential_type": "oauth", "token": "abc", "refresh_token": "def"}
        expected = OauthSecret(token="abc", refresh_token="def")
        assert secrets_factory(secret) == expected

    def test_extra_data(self):
        secret = {"credential_type": "oauth", "token": "abc", "refresh_token": "def", "ooops": "x"}
        expected = OauthSecret(token="abc", refresh_token="def")
        assert secrets_factory(secret) == expected

    def test_is_partial_secret(self):
        assert not OauthSecret(token="a", refresh_token="b").is_partial_secret()


class TestS3Secret:
    def test_minimal_data(self):
        secret = dict(credential_type="s3",)
        expected = S3Secret()
        assert secrets_factory(secret) == expected

    def test_full_data(self):
        secret = dict(
            credential_type="s3",
            aws_access_key_id="abc",
            aws_secret_access_key="abc",
            aws_session_token="abc",
            config_id="abc",
        )
        expected = S3Secret(
            aws_access_key_id="abc",
            aws_secret_access_key="abc",
            aws_session_token="abc",
            config_id="abc",
        )
        assert secrets_factory(secret) == expected

    def test_extra_data(self):
        secret = dict(credential_type="s3", oops="x",)
        expected = S3Secret()
        assert secrets_factory(secret) == expected

    def test_is_partial_secret_false(self):
        assert not S3Secret(config_id=None).is_partial_secret()

    def test_is_partial_secret_true(self):
        assert S3Secret(config_id="abc").is_partial_secret()


class TestAzureSecret:
    def test_minimal_data(self):
        secret = {
            "credential_type": "azure",
            "azure_connection_string": "abc",
        }
        expected = AzureSecret(azure_connection_string="abc")
        assert secrets_factory(secret) == expected

    def test_extra_data(self):
        secret = {"credential_type": "azure", "azure_connection_string": "abc", "ooops": "x"}
        expected = AzureSecret(azure_connection_string="abc")
        assert secrets_factory(secret) == expected

    def test_is_partial_secret(self):
        assert not AzureSecret(azure_connection_string="a").is_partial_secret()


class TestAzureServicePrincipalSecret:
    def test_minimal_data(self):
        secret = {
            "credential_type": "azure_service_principal",
            "client_id": "abc",
            "client_secret": "abc",
            "azure_tenant_id": "abc",
        }
        expected = AzureServicePrincipalSecret(
            client_id="abc", client_secret="abc", azure_tenant_id="abc",
        )
        assert secrets_factory(secret) == expected

    def test_extra_data(self):
        secret = {
            "credential_type": "azure_service_principal",
            "client_id": "abc",
            "client_secret": "abc",
            "azure_tenant_id": "abc",
            "oops": "abc",
        }
        expected = AzureServicePrincipalSecret(
            client_id="abc", client_secret="abc", azure_tenant_id="abc",
        )
        assert secrets_factory(secret) == expected

    def test_is_partial_secret(self):
        secret = AzureServicePrincipalSecret(
            client_id="abc", client_secret="abc", azure_tenant_id="abc",
        )
        assert not secret.is_partial_secret()


class TestSnowflakeOauthUserAccountSecret:
    def test_minimal_data(self):
        secret = dict(
            credential_type="snowflake_oauth_user_account",
            client_id="abc",
            client_secret="abc",
            snowflake_account_name="abc",
        )
        expected = SnowflakeOauthUserAccountSecret(
            client_id="abc", client_secret="abc", snowflake_account_name="abc",
        )
        assert secrets_factory(secret) == expected

    def test_full_data(self):
        secret = dict(
            credential_type="snowflake_oauth_user_account",
            client_id="abc",
            client_secret="abc",
            snowflake_account_name="abc",
            oauth_issuer_type="abc",
            oauth_issuer_url="abc",
            oauth_scopes="abc",
            oauth_config_id="abc",
        )
        expected = SnowflakeOauthUserAccountSecret(
            client_id="abc",
            client_secret="abc",
            snowflake_account_name="abc",
            oauth_issuer_type="abc",
            oauth_issuer_url="abc",
            oauth_scopes="abc",
            oauth_config_id="abc",
        )
        assert secrets_factory(secret) == expected

    def test_extra_data(self):
        secret = dict(
            credential_type="snowflake_oauth_user_account",
            client_id="abc",
            client_secret="abc",
            snowflake_account_name="abc",
            ooops="x",
        )
        expected = SnowflakeOauthUserAccountSecret(
            client_id="abc", client_secret="abc", snowflake_account_name="abc",
        )
        assert secrets_factory(secret) == expected

    def test_is_partial_secret_false(self):
        secret = SnowflakeOauthUserAccountSecret(
            client_id="abc",
            client_secret="abc",
            snowflake_account_name="abc",
            oauth_config_id=None,
        )
        assert not secret.is_partial_secret()

    def test_is_partial_secret_true(self):
        secret = SnowflakeOauthUserAccountSecret(
            client_id="abc",
            client_secret="abc",
            snowflake_account_name="abc",
            oauth_config_id="abc",
        )
        assert secret.is_partial_secret()


class TestSnowflakeKeyPairUserAccountSecret:
    def test_minimal_data(self):
        secret = dict(
            credential_type="snowflake_key_pair_user_account",
            username="abc",
            private_key_str="abc",
        )
        expected = SnowflakeKeyPairUserAccountSecret(username="abc", private_key_str="abc",)
        assert secrets_factory(secret) == expected

    def test_full_data(self):
        secret = dict(
            credential_type="snowflake_key_pair_user_account",
            username="abc",
            private_key_str="abc",
            passphrase="abc",
            config_id="abc",
        )
        expected = SnowflakeKeyPairUserAccountSecret(
            username="abc", private_key_str="abc", passphrase="abc", config_id="abc",
        )
        assert secrets_factory(secret) == expected

    def test_extra_data(self):
        secret = dict(
            credential_type="snowflake_key_pair_user_account",
            username="abc",
            private_key_str="abc",
            ooops="x",
        )
        expected = SnowflakeKeyPairUserAccountSecret(username="abc", private_key_str="abc",)
        assert secrets_factory(secret) == expected

    def test_is_partial_secret_false(self):
        secret = SnowflakeKeyPairUserAccountSecret(
            username="abc", private_key_str="abc", config_id=None,
        )
        assert not secret.is_partial_secret()

    def test_is_partial_secret_true(self):
        secret = SnowflakeKeyPairUserAccountSecret(
            username="abc", private_key_str="abc", config_id="abc",
        )
        assert secret.is_partial_secret()


class TestAdlsGen2OauthSecret:
    def test_data(self):
        secret = dict(
            credential_type="adls_gen2_oauth",
            client_id="abc",
            client_secret="abc",
            oauth_scopes="abc",
        )
        expected = AdlsGen2OauthSecret(client_id="abc", client_secret="abc", oauth_scopes="abc",)
        assert secrets_factory(secret) == expected

    def test_extra_data(self):
        secret = dict(
            credential_type="adls_gen2_oauth",
            client_id="abc",
            client_secret="abc",
            oauth_scopes="abc",
            oops="x",
        )
        expected = AdlsGen2OauthSecret(client_id="abc", client_secret="abc", oauth_scopes="abc",)
        assert secrets_factory(secret) == expected

    def test_is_partial_secret_false(self):
        secret = AdlsGen2OauthSecret(client_id="abc", client_secret="abc", oauth_scopes="abc",)
        assert not secret.is_partial_secret()


class TestTableauAccessTokenSecret:
    def test_data(self):
        secret = dict(
            credential_type="tableau_access_token", token_name="abc", personal_access_token="abc",
        )
        expected = TableauAccessTokenSecret(token_name="abc", personal_access_token="abc",)
        assert secrets_factory(secret) == expected

    def test_extra_data(self):
        secret = dict(
            credential_type="tableau_access_token",
            token_name="abc",
            personal_access_token="abc",
            oops="x",
        )
        expected = TableauAccessTokenSecret(token_name="abc", personal_access_token="abc",)
        assert secrets_factory(secret) == expected

    def test_is_partial_secret_false(self):
        secret = TableauAccessTokenSecret(token_name="abc", personal_access_token="abc",)
        assert not secret.is_partial_secret()
