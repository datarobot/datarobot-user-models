#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import logging
import sys
from contextlib import contextmanager
from io import StringIO
from logging import getLogger, StreamHandler, DEBUG

import pytest

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
    DatabricksAccessTokenAccountSecret,
    ApiTokenSecret,
    UnsupportedSecretError,
    load_secrets,
    patch_outputs_to_scrub_secrets,
    get_ordered_sensitive_values,
    scrub_values_from_string,
    TextStreamSecretsScrubber,
    SecretsScrubberFilter,
    reset_outputs_to_allow_secrets,
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
        expected_key = GCPKey(
            "abc",
        )
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
        expected_key = GCPKey(
            "abc",
            "abc",
            "abc",
            "abc",
            "abc",
            "abc",
            "abc",
            "abc",
            "abc",
            "abc",
        )
        secret = {"credential_type": "gcp", "gcp_key": gcp_key}
        expected = GCPSecret(gcp_key=expected_key)
        assert secrets_factory(secret) == expected

    def test_repr_and_str(self):
        gcp_secret = GCPSecret(gcp_key=GCPKey(type="abc", project_id="xyt"))
        expected = (
            "GCPSecret(gcp_key=GCPKey("
            "type='*****', project_id='*****', private_key_id=None, private_key=None, client_email=None, "
            "client_id=None, auth_uri=None, token_uri=None, "
            "auth_provider_x509_cert_url=None, client_x509_cert_url=None), config_id=None)"
        )
        assert repr(gcp_secret) == str(gcp_secret) == expected


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

    def test_repr_and_str(self):
        secret = BasicSecret(username="x", password="y")
        expected = "BasicSecret(username='*****', password='*****', snowflake_account_name=None)"
        actual = repr(secret)
        assert actual == str(secret) == expected


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

    def test_str_and_repr(self):
        secret = OauthSecret(token="a", refresh_token="b")
        expected = "OauthSecret(token='*****', refresh_token='*****')"
        assert str(secret) == repr(secret) == expected


class TestS3Secret:
    def test_minimal_data(self):
        secret = dict(
            credential_type="s3",
        )
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
        secret = dict(
            credential_type="s3",
            oops="x",
        )
        expected = S3Secret()
        assert secrets_factory(secret) == expected

    def test_is_partial_secret_false(self):
        assert not S3Secret(config_id=None).is_partial_secret()

    def test_is_partial_secret_true(self):
        assert S3Secret(config_id="abc").is_partial_secret()

    def test_repr_and_str(self):
        secret = S3Secret(aws_secret_access_key="abc")
        expected = "S3Secret(aws_access_key_id=None, aws_secret_access_key='*****', aws_session_token=None, config_id=None)"
        assert str(secret) == repr(secret) == expected


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

    def test_repr_and_str(self):
        secret = AzureSecret(azure_connection_string="a")
        expected = "AzureSecret(azure_connection_string='*****')"
        assert str(secret) == repr(secret) == expected


class TestAzureServicePrincipalSecret:
    def test_minimal_data(self):
        secret = {
            "credential_type": "azure_service_principal",
            "client_id": "abc",
            "client_secret": "abc",
            "azure_tenant_id": "abc",
        }
        expected = AzureServicePrincipalSecret(
            client_id="abc",
            client_secret="abc",
            azure_tenant_id="abc",
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
            client_id="abc",
            client_secret="abc",
            azure_tenant_id="abc",
        )
        assert secrets_factory(secret) == expected

    def test_is_partial_secret(self):
        secret = AzureServicePrincipalSecret(
            client_id="abc",
            client_secret="abc",
            azure_tenant_id="abc",
        )
        assert not secret.is_partial_secret()

    def test_repr_and_str(self):
        secret = AzureServicePrincipalSecret(
            client_id="asdkfjslkf",
            client_secret="asdkjhdfj",
            azure_tenant_id="x",
        )
        expected = "AzureServicePrincipalSecret(client_id='*****', client_secret='*****', azure_tenant_id='*****')"

        assert str(secret) == repr(secret) == expected


class TestSnowflakeOauthUserAccountSecret:
    def test_minimal_data(self):
        secret = dict(
            credential_type="snowflake_oauth_user_account",
            client_id="abc",
            client_secret="abc",
            snowflake_account_name="abc",
        )
        expected = SnowflakeOauthUserAccountSecret(
            client_id="abc",
            client_secret="abc",
            snowflake_account_name="abc",
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
            client_id="abc",
            client_secret="abc",
            snowflake_account_name="abc",
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

    def test_repr_and_str(self):
        secret = SnowflakeOauthUserAccountSecret(
            client_id="abc",
            client_secret="a",
            snowflake_account_name="xyz",
        )
        expected = (
            "SnowflakeOauthUserAccountSecret(client_id='*****', client_secret='*****', "
            "snowflake_account_name='*****', oauth_issuer_type=None, oauth_issuer_url=None, oauth_scopes=None, "
            "oauth_config_id=None)"
        )

        assert str(secret) == repr(secret) == expected


class TestSnowflakeKeyPairUserAccountSecret:
    def test_minimal_data(self):
        secret = dict(
            credential_type="snowflake_key_pair_user_account",
            username="abc",
            private_key_str="abc",
        )
        expected = SnowflakeKeyPairUserAccountSecret(
            username="abc",
            private_key_str="abc",
        )
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
            username="abc",
            private_key_str="abc",
            passphrase="abc",
            config_id="abc",
        )
        assert secrets_factory(secret) == expected

    def test_extra_data(self):
        secret = dict(
            credential_type="snowflake_key_pair_user_account",
            username="abc",
            private_key_str="abc",
            ooops="x",
        )
        expected = SnowflakeKeyPairUserAccountSecret(
            username="abc",
            private_key_str="abc",
        )
        assert secrets_factory(secret) == expected

    def test_is_partial_secret_false(self):
        secret = SnowflakeKeyPairUserAccountSecret(
            username="abc",
            private_key_str="abc",
            config_id=None,
        )
        assert not secret.is_partial_secret()

    def test_is_partial_secret_true(self):
        secret = SnowflakeKeyPairUserAccountSecret(
            username="abc",
            private_key_str="abc",
            config_id="abc",
        )
        assert secret.is_partial_secret()

    def test_repr_and_str(self):
        secret = SnowflakeKeyPairUserAccountSecret(
            username="abc",
            private_key_str="Key",
        )

        expected = (
            "SnowflakeKeyPairUserAccountSecret(username='*****', private_key_str='*****', "
            "passphrase=None, config_id=None)"
        )

        assert str(secret) == repr(secret) == expected


class TestAdlsGen2OauthSecret:
    def test_data(self):
        secret = dict(
            credential_type="adls_gen2_oauth",
            client_id="abc",
            client_secret="abc",
            oauth_scopes="abc",
        )
        expected = AdlsGen2OauthSecret(
            client_id="abc",
            client_secret="abc",
            oauth_scopes="abc",
        )
        assert secrets_factory(secret) == expected

    def test_extra_data(self):
        secret = dict(
            credential_type="adls_gen2_oauth",
            client_id="abc",
            client_secret="abc",
            oauth_scopes="abc",
            oops="x",
        )
        expected = AdlsGen2OauthSecret(
            client_id="abc",
            client_secret="abc",
            oauth_scopes="abc",
        )
        assert secrets_factory(secret) == expected

    def test_is_partial_secret_false(self):
        secret = AdlsGen2OauthSecret(
            client_id="abc",
            client_secret="abc",
            oauth_scopes="abc",
        )
        assert not secret.is_partial_secret()

    def test_repr_and_str(self):
        secret = AdlsGen2OauthSecret(
            client_id="abc",
            client_secret="sdsdf",
            oauth_scopes="svfj",
        )
        expected = (
            "AdlsGen2OauthSecret(client_id='*****', client_secret='*****', oauth_scopes='*****')"
        )

        assert str(secret) == repr(secret) == expected


class TestTableauAccessTokenSecret:
    def test_data(self):
        secret = dict(
            credential_type="tableau_access_token",
            token_name="abc",
            personal_access_token="abc",
        )
        expected = TableauAccessTokenSecret(
            token_name="abc",
            personal_access_token="abc",
        )
        assert secrets_factory(secret) == expected

    def test_extra_data(self):
        secret = dict(
            credential_type="tableau_access_token",
            token_name="abc",
            personal_access_token="abc",
            oops="x",
        )
        expected = TableauAccessTokenSecret(
            token_name="abc",
            personal_access_token="abc",
        )
        assert secrets_factory(secret) == expected

    def test_is_partial_secret_false(self):
        secret = TableauAccessTokenSecret(
            token_name="abc",
            personal_access_token="abc",
        )
        assert not secret.is_partial_secret()

    def test_repr_and_str(self):
        secret = TableauAccessTokenSecret(
            token_name="abc",
            personal_access_token="abc",
        )
        expected = "TableauAccessTokenSecret(token_name='*****', personal_access_token='*****')"

        assert str(secret) == repr(secret) == expected


class TestDatabricksAccessTokenAccountSecret:
    def test_data(self):
        secret = dict(
            credential_type="databricks_access_token_account",
            databricks_access_token="abc",
        )
        expected = DatabricksAccessTokenAccountSecret(
            databricks_access_token="abc",
        )
        assert secrets_factory(secret) == expected

    def test_extra_data(self):
        secret = dict(
            credential_type="databricks_access_token_account",
            databricks_access_token="abc",
            oops="x",
        )
        expected = DatabricksAccessTokenAccountSecret(
            databricks_access_token="abc",
        )
        assert secrets_factory(secret) == expected

    def test_is_partial_secret_false(self):
        secret = DatabricksAccessTokenAccountSecret(
            databricks_access_token="abc",
        )
        assert not secret.is_partial_secret()

    def test_repr_and_str(self):
        secret = DatabricksAccessTokenAccountSecret(
            databricks_access_token="abc",
        )
        expected = "DatabricksAccessTokenAccountSecret(databricks_access_token='*****')"

        assert str(secret) == repr(secret) == expected


class TestApiTokenSecret:
    def test_data(self):
        secret = dict(
            credential_type="api_token",
            api_token="abc",
        )
        expected = ApiTokenSecret(
            api_token="abc",
        )
        assert secrets_factory(secret) == expected

    def test_extra_data(self):
        secret = dict(
            credential_type="api_token",
            api_token="abc",
            oops="x",
        )
        expected = ApiTokenSecret(
            api_token="abc",
        )
        assert secrets_factory(secret) == expected

    def test_is_partial_secret_false(self):
        secret = ApiTokenSecret(
            api_token="abc",
        )
        assert not secret.is_partial_secret()

    def test_repr_and_str(self):
        secret = ApiTokenSecret(
            api_token="abc",
        )
        expected = "ApiTokenSecret(api_token='*****')"

        assert str(secret) == repr(secret) == expected


class TestUnsupportedSecret:
    def test_unsupported_secret_type(self):
        bad_type = "wuuuuuuuut"
        with pytest.raises(UnsupportedSecretError, match=f"type: {bad_type!r}"):
            secrets_factory({"credential_type": bad_type})


class TestLoadSecrets:
    def test_load_secrets_no_file_no_env_vars(self):
        secrets = load_secrets(None, None)
        assert secrets == {}

    def test_load_secrets_mount_path_does_not_exist(self):
        bad_dir = "/nope/not/a/thing"
        secrets = load_secrets(bad_dir, None)
        assert secrets == {}

    def test_secrets_with_mounted_secrets(self, mounted_secrets_factory):
        secrets = {
            "ONE": {"credential_type": "basic", "username": "1", "password": "y"},
            "TWO": {"credential_type": "basic", "username": "2", "password": "z"},
        }
        secrets_dir = mounted_secrets_factory(secrets)

        expected_secrets = {
            "ONE": BasicSecret(username="1", password="y"),
            "TWO": BasicSecret(username="2", password="z"),
        }

        assert load_secrets(secrets_dir, None) == expected_secrets

    def test_secrets_with_env_vars(self, env_patcher):
        secrets = {
            "ONE": {"credential_type": "basic", "username": "1", "password": "y"},
            "TWO": {"credential_type": "basic", "username": "2", "password": "z"},
        }
        prefix = "MY_SUPER_PREFIX"

        expected_secrets = {
            "ONE": BasicSecret(username="1", password="y"),
            "TWO": BasicSecret(username="2", password="z"),
        }
        with env_patcher(prefix, secrets):
            assert load_secrets(None, prefix) == expected_secrets

    def test_secrets_with_mounted_secrets_supersede_env_secrets(
        self, mounted_secrets_factory, env_patcher
    ):
        mounted_secrets = {
            "ONE": {"credential_type": "basic", "username": "1", "password": "y"},
            "TWO": {"credential_type": "basic", "username": "2", "password": "z"},
        }
        env_secrets = {
            "TWO": {"credential_type": "basic", "username": "superseded", "password": "superseded"},
            "THREE": {"credential_type": "basic", "username": "3", "password": "A"},
        }
        prefix = "MY_SUPER_PREFIX"
        secrets_dir = mounted_secrets_factory(mounted_secrets)

        expected_secrets = {
            "ONE": BasicSecret(username="1", password="y"),
            "TWO": BasicSecret(username="2", password="z"),
            "THREE": BasicSecret(username="3", password="A"),
        }
        with env_patcher(prefix, env_secrets):
            assert load_secrets(secrets_dir, prefix) == expected_secrets

    @pytest.mark.parametrize(
        "secret_dict, expected",
        [
            ({"credential_type": "gcp", "gcp_key": None}, GCPSecret(None)),
            ({"credential_type": "basic", "username": "a", "password": "b"}, BasicSecret("a", "b")),
            (
                {"credential_type": "oauth", "token": "a", "refresh_token": "b"},
                OauthSecret("a", "b"),
            ),
            (dict(credential_type="s3"), S3Secret()),
            (dict(credential_type="azure", azure_connection_string="a"), AzureSecret("a")),
            (
                dict(
                    credential_type="azure_service_principal",
                    client_id="a",
                    client_secret="a",
                    azure_tenant_id="a",
                ),
                AzureServicePrincipalSecret("a", "a", "a"),
            ),
            (
                dict(
                    credential_type="snowflake_oauth_user_account",
                    client_id="a",
                    client_secret="a",
                    snowflake_account_name="a",
                ),
                SnowflakeOauthUserAccountSecret("a", "a", "a"),
            ),
            (
                dict(
                    credential_type="snowflake_key_pair_user_account",
                    username="a",
                    private_key_str="a",
                ),
                SnowflakeKeyPairUserAccountSecret("a", "a"),
            ),
            (
                dict(
                    credential_type="adls_gen2_oauth",
                    client_id="a",
                    client_secret="a",
                    oauth_scopes="a",
                ),
                AdlsGen2OauthSecret("a", "a", "a"),
            ),
            (
                dict(
                    credential_type="tableau_access_token",
                    token_name="a",
                    personal_access_token="a",
                ),
                TableauAccessTokenSecret("a", "a"),
            ),
            (
                dict(
                    credential_type="databricks_access_token_account",
                    databricks_access_token="a",
                ),
                DatabricksAccessTokenAccountSecret("a"),
            ),
            (dict(credential_type="api_token", api_token="a"), ApiTokenSecret("a")),
        ],
    )
    def test_load_all_secret_types(self, secret_dict, expected, env_patcher):
        env_dict = {"a": secret_dict}
        expected_dict = {"a": expected}
        with env_patcher("PREFIX", env_dict):
            assert load_secrets(None, "PREFIX") == expected_dict


def get_content(stream):
    stream.seek(0)
    return stream.read()


class TestGetOrderedSensitiveValues:
    def test_no_secrets(self):
        assert get_ordered_sensitive_values([]) == []

    @pytest.mark.parametrize(
        "secret, expected",
        [
            (BasicSecret(username="a", password="b"), ["a", "b"]),
            (BasicSecret(username="c", password="b", snowflake_account_name="a"), ["a", "b", "c"]),
            (OauthSecret(token="abc", refresh_token="def"), ["abc", "def"]),
        ],
    )
    def test_flat_secret(self, secret, expected):
        assert get_ordered_sensitive_values([secret]) == expected

    def test_nested_secret(self):
        secret = GCPSecret(gcp_key=GCPKey(type="a", project_id="d", private_key="c"), config_id="b")
        assert get_ordered_sensitive_values([secret]) == ["a", "b", "c", "d"]

    def test_sorts_by_largest_size_first(self):
        secret = BasicSecret(username="de", password="xyz", snowflake_account_name="abs")
        assert get_ordered_sensitive_values([secret]) == ["abs", "xyz", "de"]

    def test_multiple_secrets(self):
        secrets = [
            BasicSecret(username="a", password="bc", snowflake_account_name="def"),
            BasicSecret(username="x", password="yz", snowflake_account_name="abc"),
        ]
        assert get_ordered_sensitive_values(secrets) == ["abc", "def", "bc", "yz", "a", "x"]

    def test_multiple_secrets_with_iterable(self):
        secrets_dict = {
            "a": BasicSecret(username="a", password="bc", snowflake_account_name="def"),
            "b": BasicSecret(username="x", password="yz", snowflake_account_name="abc"),
        }
        assert get_ordered_sensitive_values(secrets_dict.values()) == [
            "abc",
            "def",
            "bc",
            "yz",
            "a",
            "x",
        ]

    def test_repeat_values(self):
        secrets = [BasicSecret(username="a", password="b"), BasicSecret(username="b", password="c")]
        assert get_ordered_sensitive_values(secrets) == ["a", "b", "c"]


class TestScrubValuesFromString:
    def test_no_sensitive_values(self):
        assert scrub_values_from_string([], "abc") == "abc"

    def test_sensitive_values_not_in_string(self):
        assert scrub_values_from_string(["abc", "def"], "ab de") == "ab de"

    def test_scrubs_values(self):
        assert scrub_values_from_string(["abc", "def"], "def abcd") == "***** *****d"

    def test_scrubs_values_order_sensitive_positive_case(self):
        assert scrub_values_from_string(["abc", "ab"], "def ab abc") == "def ***** *****"

    def test_scrubs_values_order_sensitive_failure_on_unsorted_values(self):
        assert scrub_values_from_string(["ab", "abc"], "def ab abc") == "def ***** *****c"


class TestTextStreamSecretsScrubber:
    def test_scrubs_output(self):
        stream = StringIO()
        wrapped = TextStreamSecretsScrubber([BasicSecret(username="abc", password="bc")], stream)
        wrapped.write("abc def ab bc\n")
        wrapped.write("bcd\n")
        assert get_content(wrapped) == "***** def ab *****\n*****d\n"

    def test_scrubs_output_multiple_secrets(self):
        stream = StringIO()
        wrapped = TextStreamSecretsScrubber(
            [ApiTokenSecret(api_token="abc"), ApiTokenSecret(api_token="bc")], stream
        )
        wrapped.write("abc def ab bc\n")
        wrapped.write("bcd\n")
        assert get_content(wrapped) == "***** def ab *****\n*****d\n"

    def test_scrubs_output_on_writelines(self):
        stream = StringIO()
        wrapped = TextStreamSecretsScrubber([BasicSecret(username="abc", password="bc")], stream)
        wrapped.writelines(["abc def ab bc\n", "bcd\n"])
        assert get_content(wrapped) == "***** def ab *****\n*****d\n"


class TestSecretsScrubberFilter:
    @pytest.fixture
    def text_stream(self):
        return StringIO()

    @pytest.fixture
    def logger(self, text_stream):
        logger = getLogger(__name__)
        logger.setLevel(DEBUG)
        logger.addHandler(StreamHandler(text_stream))
        yield logger
        for added_filter in logger.filters:
            logger.removeFilter(added_filter)

    def test_setup(self, text_stream, logger):
        logger.info("hello")
        logger.error("hi")
        assert get_content(text_stream) == "hello\nhi\n"

    def test_single_secret_msg(self, logger, text_stream):
        secrets_filter = SecretsScrubberFilter([BasicSecret(username="def", password="ef")])
        logger.addFilter(secrets_filter)
        logger.info("abc def")
        logger.info("ef def")
        assert get_content(text_stream) == "abc *****\n***** *****\n"

    def test_multiple_secrets_msg(self, logger, text_stream):
        secrets_filter = SecretsScrubberFilter(
            [ApiTokenSecret(api_token="def"), ApiTokenSecret(api_token="ef")]
        )
        logger.addFilter(secrets_filter)
        logger.info("abc def")
        logger.info("ef def")
        assert get_content(text_stream) == "abc *****\n***** *****\n"

    def test_secret_passed_to_msg(self, logger, text_stream):
        secret = ApiTokenSecret(api_token="def")
        secrets_filter = SecretsScrubberFilter([secret])
        logger.addFilter(secrets_filter)
        logger.info(secret)

        expected = f"{secret}\n".replace("def", "*****")
        assert get_content(text_stream) == expected

    def test_other_object_passed_to_msg(self, logger, text_stream):
        secrets_filter = SecretsScrubberFilter([(ApiTokenSecret(api_token="def"))])
        logger.addFilter(secrets_filter)
        something = object()
        logger.info(something)

        expected = f"{something}\n"
        assert get_content(text_stream) == expected

    def test_args_single_arg_string(self, logger, text_stream):
        secrets_filter = SecretsScrubberFilter([(ApiTokenSecret(api_token="xyz"))])
        logger.addFilter(secrets_filter)
        logger.info("hello %s", "xyz")

        expected = "hello *****\n"
        assert get_content(text_stream) == expected

    def test_args_single_arg_secret_object(self, logger, text_stream):
        secret = ApiTokenSecret(api_token="xyz")
        secrets_filter = SecretsScrubberFilter([secret])
        logger.addFilter(secrets_filter)
        logger.info("hello %s", secret)

        expected = f"hello {secret}\n".replace("xyz", "*****")
        assert get_content(text_stream) == expected

    def test_args_single_arg_other_object(self, logger, text_stream):
        secrets_filter = SecretsScrubberFilter([(ApiTokenSecret(api_token="xyz"))])
        logger.addFilter(secrets_filter)
        something = object()
        logger.info("hello %s", something)

        expected = f"hello {something}\n"
        assert get_content(text_stream) == expected

    def test_args_multiple_args_string(self, logger, text_stream):
        secrets_filter = SecretsScrubberFilter([(ApiTokenSecret(api_token="xyz"))])
        logger.addFilter(secrets_filter)
        logger.info("hello %s %d", "xyz", 1)

        expected = "hello ***** 1\n"
        assert get_content(text_stream) == expected

    def test_args_multiple_args_secret_object(self, logger, text_stream):
        secret = ApiTokenSecret(api_token="xyz")
        secrets_filter = SecretsScrubberFilter([secret])
        logger.addFilter(secrets_filter)
        logger.info("hello %s %d", secret, 2)

        expected = f"hello {secret} 2\n".replace("xyz", "*****")
        assert get_content(text_stream) == expected

    def test_args_multiple_args_other_object(self, logger, text_stream):
        secrets_filter = SecretsScrubberFilter([(ApiTokenSecret(api_token="xyz"))])
        logger.addFilter(secrets_filter)
        something = object()
        logger.info("hello %s %d", something, 3)

        expected = f"hello {something} 3\n"
        assert get_content(text_stream) == expected

    def test_dict_args_string(self, logger, text_stream):
        secrets_filter = SecretsScrubberFilter([(ApiTokenSecret(api_token="xyz"))])
        logger.addFilter(secrets_filter)
        logger.info("hello %(a)s %(b)d", {"a": "xyz", "b": 1})

        expected = "hello ***** 1\n"
        assert get_content(text_stream) == expected

    def test_dict_args_secret_object(self, logger, text_stream):
        secret = ApiTokenSecret(api_token="xyz")
        secrets_filter = SecretsScrubberFilter([secret])
        logger.addFilter(secrets_filter)
        logger.info("hello %(a)s %(b)d", {"a": secret, "b": 2})

        expected = f"hello {secret} 2\n".replace("xyz", "*****")
        assert get_content(text_stream) == expected

    def test_dict_args_other_object(self, logger, text_stream):
        secrets_filter = SecretsScrubberFilter([(ApiTokenSecret(api_token="xyz"))])
        logger.addFilter(secrets_filter)
        something = object()
        logger.info("hello %(a)s %(b)d", {"a": something, "b": 3})

        expected = f"hello {something} 3\n"
        assert get_content(text_stream) == expected


@contextmanager
def stdout_context():
    new_stdout = StringIO()
    old_stdout = sys.stdout

    try:
        sys.stdout = new_stdout
        yield new_stdout

    finally:
        sys.stdout = old_stdout


@contextmanager
def stderr_context():
    new_stderr = StringIO()
    old_stderr = sys.stderr

    try:
        sys.stderr = new_stderr
        yield new_stderr

    finally:
        sys.stderr = old_stderr


@contextmanager
def contextualized_patch_outputs_to_scrub_secrets(secrets):
    try:
        patch_outputs_to_scrub_secrets(secrets)
        yield

    finally:
        if isinstance(sys.stdout, TextStreamSecretsScrubber):
            sys.stdout = sys.stdout.stream
        if isinstance(sys.stderr, TextStreamSecretsScrubber):
            sys.stderr = sys.stderr.stream

        for logger in logging.root.manager.loggerDict.values():
            _remove_scrubber_filters(logger)
        _remove_scrubber_filters(logging.root)


def _remove_scrubber_filters(logger):
    if not hasattr(logger, "filters"):
        return
    for logging_filter in logger.filters:
        if isinstance(logging_filter, SecretsScrubberFilter):
            logger.removeFilter(logging_filter)


class TestPatchOutputToScrubSecrets:
    @pytest.fixture
    def secrets(self):
        yield [ApiTokenSecret(api_token="ab"), BasicSecret(username="cd", password="ef")]

    def test_testing_setup_stdout(self):
        with stdout_context() as mock_out:
            print("hello")
        assert get_content(mock_out) == "hello\n"

    def test_testing_setup_stderr(self):
        with stderr_context() as mock_err:
            print("hello", file=sys.stderr)
        assert get_content(mock_err) == "hello\n"

    def test_basic_secret_scrubbing_stdout(self):
        scrub_one = "delme"
        scrub_two = "and me"
        secrets = [BasicSecret(username=scrub_one, password=scrub_two)]
        with stdout_context() as mock_out:
            with contextualized_patch_outputs_to_scrub_secrets(secrets):
                print(f"x{scrub_one}y{scrub_two}z")
        expected = "x*****y*****z\n"
        assert get_content(mock_out) == expected

    def test_basic_secret_scrubbing_stderr(self):
        scrub_one = "delme"
        scrub_two = "and me"
        secrets = [BasicSecret(username=scrub_one, password=scrub_two)]
        with stderr_context() as mock_err:
            with contextualized_patch_outputs_to_scrub_secrets(secrets):
                print(f"x{scrub_one}y{scrub_two}z", file=sys.stderr)
        expected = "x*****y*****z\n"
        assert get_content(mock_err) == expected

    def test_no_secrets_does_not_patch_stdout_stderr(self):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        with contextualized_patch_outputs_to_scrub_secrets([]):
            assert sys.stderr == original_stderr
            assert sys.stdout == original_stdout
        assert sys.stdout == original_stdout
        assert sys.stderr == original_stderr

    def test_patches_stdout(self):
        original_stream = sys.stdout
        secrets = [BasicSecret(username="abc", password="def")]

        expected = TextStreamSecretsScrubber(secrets, original_stream)

        with contextualized_patch_outputs_to_scrub_secrets(secrets):
            assert sys.stdout == expected
        assert sys.stdout == original_stream

    def test_patches_stderr(self):
        original_stream = sys.stderr
        secrets = [BasicSecret(username="zyx", password="ghi")]

        expected = TextStreamSecretsScrubber(secrets, original_stream)

        with contextualized_patch_outputs_to_scrub_secrets(secrets):
            assert sys.stderr == expected
        assert sys.stderr == original_stream

    def test_no_secrets_does_not_patch_loggers(self):
        root_logger = logging.root.manager.root
        other_logger = [getLogger("a"), getLogger("a.b")]
        with contextualized_patch_outputs_to_scrub_secrets([]):
            other_filters = [el.filters[:] for el in other_logger]
            root_filters = root_logger.filters[:]
        assert other_filters == [[], []]
        assert all(not isinstance(el, SecretsScrubberFilter) for el in root_filters)

    def test_patches_root_logging_methods(self):
        stream = StringIO()
        root_logger = logging.root.manager.root
        root_logger.addHandler(StreamHandler(stream=stream))

        secrets = [ApiTokenSecret(api_token="hello")]
        with contextualized_patch_outputs_to_scrub_secrets(secrets):
            logging.error("hello there")
            filters = root_logger.filters[:]

        assert get_content(stream) == "***** there\n"
        assert SecretsScrubberFilter(secrets) in filters

    def test_patches_all_loggers(self):
        loggers = [
            getLogger("a"),
            getLogger("b"),
            getLogger("a.b"),
            getLogger("a.c"),
            getLogger("a.b.c"),
        ]

        secrets = [ApiTokenSecret(api_token="hello")]
        with contextualized_patch_outputs_to_scrub_secrets(secrets):
            logging.error("hello there")
            filters_map = {logger.name: logger.filters[:] for logger in loggers}

        expected = [SecretsScrubberFilter(secrets)]
        for logger_name, filters in filters_map.items():
            assert filters == expected, f"Logger: {logger_name!r}"


class TestResetOutputsToAllowSecrets:
    def test_un_patches_stdout(self):
        original_stream = sys.stdout
        secrets = [BasicSecret(username="abc", password="def")]

        with contextualized_patch_outputs_to_scrub_secrets(secrets):
            reset_outputs_to_allow_secrets()
            assert sys.stdout == original_stream
        assert sys.stdout == original_stream

    def test_un_patches_stderr(self):
        original_stream = sys.stderr
        secrets = [BasicSecret(username="zyx", password="ghi")]

        with contextualized_patch_outputs_to_scrub_secrets(secrets):
            reset_outputs_to_allow_secrets()
            assert sys.stderr == original_stream
        assert sys.stderr == original_stream

    def test_un_filters_root_logger(self):
        root_logger = logging.root.manager.root
        original_filters = root_logger.filters[:]
        secrets = [ApiTokenSecret(api_token="hello")]

        expected_not_present = SecretsScrubberFilter(secrets)
        assert expected_not_present not in original_filters

        with contextualized_patch_outputs_to_scrub_secrets(secrets):
            reset_outputs_to_allow_secrets()
            assert expected_not_present not in root_logger.filters
            assert root_logger.filters == original_filters

    def test_un_filters_all_loggers(self):
        loggers = [
            getLogger("a"),
            getLogger("b"),
            getLogger("a.b"),
            getLogger("a.c"),
            getLogger("a.b.c"),
        ]

        original_filters_map = {el.name: el.filters[:] for el in loggers}

        secrets = [ApiTokenSecret(api_token="hello")]
        for logger_name, filters in original_filters_map.items():
            assert filters == [], f"Logger: {logger_name!r}"

        with contextualized_patch_outputs_to_scrub_secrets(secrets):
            reset_outputs_to_allow_secrets()
            current_filters_map = {logger.name: logger.filters[:] for logger in loggers}
            for logger_name, filters in current_filters_map.items():
                assert filters == [], f"Logger: {logger_name!r}"
