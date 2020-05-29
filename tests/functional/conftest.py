import os
import pytest
import uuid
import warnings
from dr_usertool.datarobot_user_database import DataRobotUserDatabase
from dr_usertool.utils import get_permissions


def dr_usertool_setup():
    mongo_host = os.environ.get("MONGO_HOST", os.environ.get("HOST", "127.0.0.1")).strip()
    return DataRobotUserDatabase.setup("adhoc", "", mongo_host=mongo_host)


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    # check for skipping setup on xdist master process
    if not config.pluginmanager.getplugin("dsession"):
        suffix = str(uuid.uuid4().int)
        env, db = dr_usertool_setup()

        # User credentials
        user_username = "local-custom-model-templates-tests-{}@datarobot.com".format(suffix)
        user_api_token = "lkjkljnm988989jkr5645tv_{}".format(suffix)
        user_permissions = get_permissions("tests/fixtures/user_permissions.json", user_api_token)

        # Add user
        DataRobotUserDatabase.add_user(
            db,
            env,
            user_username,
            invite_code="autogen",
            app_user_manager=False,
            permissions=user_permissions,
            api_token=user_api_token,
            activated=True,
            unix_user="datarobot_imp",
        )
        os.environ["DATAROBOT_API_TOKEN"] = user_api_token
        config.user_username = user_username


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    if not config.pluginmanager.getplugin("dsession"):
        warnings.simplefilter("ignore")
        _, db = dr_usertool_setup()
        DataRobotUserDatabase.delete_user(db, config.user_username)
        warnings.simplefilter("error")
