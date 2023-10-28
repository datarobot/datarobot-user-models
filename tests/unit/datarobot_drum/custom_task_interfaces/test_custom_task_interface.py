#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#

from datarobot_drum.custom_task_interfaces.custom_task_interface import (
    CustomTaskInterface,
    secrets_injection_context,
)
from datarobot_drum.custom_task_interfaces.user_secrets import BasicSecret


class TestSecretsInjectionContext:
    def test_default_empty_secrets(self):
        interface = CustomTaskInterface()
        assert interface.secrets == {}

    def test_load_secrets_no_file_no_env_vars(self):
        interface = CustomTaskInterface()
        with secrets_injection_context(interface, None, None):
            assert interface.secrets == {}
        assert interface.secrets == {}

    def test_load_secrets_mount_path_does_not_exist(self):
        interface = CustomTaskInterface()
        bad_dir = "/nope/not/a/thing"
        with secrets_injection_context(interface, bad_dir, None):
            assert interface.secrets == {}
        assert interface.secrets == {}

    def test_secrets_with_mounted_secrets(self, mounted_secrets_factory):
        secrets = {
            "ONE": {"credential_type": "basic", "username": "1", "password": "y"},
            "TWO": {"credential_type": "basic", "username": "2", "password": "z"},
        }
        secrets_dir = mounted_secrets_factory(secrets)
        interface = CustomTaskInterface()

        expected_secrets = {
            "ONE": BasicSecret(username="1", password="y"),
            "TWO": BasicSecret(username="2", password="z"),
        }

        with secrets_injection_context(interface, secrets_dir, None):
            assert interface.secrets == expected_secrets
        assert not interface.secrets

    def test_secrets_with_env_vars(self, env_patcher):
        secrets = {
            "ONE": {"credential_type": "basic", "username": "1", "password": "y"},
            "TWO": {"credential_type": "basic", "username": "2", "password": "z"},
        }
        prefix = "MY_SUPER_PREFIX"
        interface = CustomTaskInterface()

        expected_secrets = {
            "ONE": BasicSecret(username="1", password="y"),
            "TWO": BasicSecret(username="2", password="z"),
        }
        with env_patcher(prefix, secrets):
            with secrets_injection_context(interface, None, prefix):
                assert interface.secrets == expected_secrets
        assert not interface.secrets

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
        interface = CustomTaskInterface()

        expected_secrets = {
            "ONE": BasicSecret(username="1", password="y"),
            "TWO": BasicSecret(username="2", password="z"),
            "THREE": BasicSecret(username="3", password="A"),
        }
        with env_patcher(prefix, env_secrets):
            with secrets_injection_context(interface, secrets_dir, prefix):
                assert interface.secrets == expected_secrets
        assert not interface.secrets
