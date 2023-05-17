"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import os
import uuid
import warnings
from urllib.parse import urlparse

import datarobot as dr
import pytest
from dr_usertool.datarobot_user_database import DataRobotUserDatabase
from dr_usertool.utils import get_permissions

from tests.drum.constants import PUBLIC_DROPIN_ENVS_PATH, TESTS_DATA_PATH

WEBSERVER_URL = "http://localhost"
ENDPOINT_URL = WEBSERVER_URL + "/api/v2"


def get_admin_api_key():
    admin_api_key = os.environ.get("APP_ADMIN_API_KEY")
    if not admin_api_key:
        raise ValueError("APP_ADMIN_API_KEY environment variable is not set")
    return admin_api_key


def dr_usertool_setup():
    webserver = urlparse(WEBSERVER_URL)
    admin_api_key = get_admin_api_key()
    return DataRobotUserDatabase.setup(
        "adhoc", webserver.hostname, protocol=webserver.scheme, admin_api_key=admin_api_key
    )


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    # check for skipping setup on xdist master process
    if not config.pluginmanager.getplugin("dsession"):
        suffix = str(uuid.uuid4())
        env = dr_usertool_setup()
        admin_api_key = get_admin_api_key()

        # User credentials
        user_username = "local-custom-model-tests-{}@datarobot.com".format(suffix)
        user_password = "Lkjkljnm988989jkr5645tv_{}".format(suffix)
        user_api_key_name = "drum-functional-tests"
        user_permissions = get_permissions(
            "tests/fixtures/user_permissions.json", user_api_key_name
        )

        # Add user
        DataRobotUserDatabase.add_user(
            env,
            user_username,
            admin_api_key=admin_api_key,
            password=user_password,
            permissions=user_permissions,
            api_key_name=user_api_key_name,
            activated=True,
            unix_user="datarobot_imp",
        )

        user_api_keys = DataRobotUserDatabase.get_api_keys(env, user_username, user_password)
        user_api_key = user_api_keys["data"][0]["key"]

        os.environ["DATAROBOT_API_TOKEN"] = user_api_key  # TODO: rename to DATAROBOT_API_KEY
        os.environ["DATAROBOT_ENDPOINT"] = ENDPOINT_URL
        config.user_username = user_username


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    if not config.pluginmanager.getplugin("dsession"):
        warnings.simplefilter("ignore")
        env = dr_usertool_setup()
        admin_api_key = get_admin_api_key()
        DataRobotUserDatabase.delete_user(env, config.user_username, admin_api_key=admin_api_key)
        warnings.simplefilter("error")


def pytest_sessionstart(session):
    dr.Client(endpoint=ENDPOINT_URL, token=os.environ["DATAROBOT_API_TOKEN"])


@pytest.fixture(scope="session")
def java_drop_in_env():
    env_dir = os.path.join(PUBLIC_DROPIN_ENVS_PATH, "java_codegen")
    environment = dr.ExecutionEnvironment.create(name="java_drop_in", programming_language="java")
    environment_version = dr.ExecutionEnvironmentVersion.create(environment.id, env_dir)
    return environment.id, environment_version.id


@pytest.fixture(scope="session")
def sklearn_drop_in_env():
    env_dir = os.path.join(PUBLIC_DROPIN_ENVS_PATH, "python3_sklearn")
    environment = dr.ExecutionEnvironment.create(
        name="python3_sklearn", programming_language="python"
    )
    environment_version = dr.ExecutionEnvironmentVersion.create(environment.id, env_dir)
    return environment.id, environment_version.id


@pytest.fixture(scope="session")
def xgboost_drop_in_env():
    env_dir = os.path.join(PUBLIC_DROPIN_ENVS_PATH, "python3_xgboost")
    environment = dr.ExecutionEnvironment.create(
        name="python3_xgboost", programming_language="python"
    )
    environment_version = dr.ExecutionEnvironmentVersion.create(environment.id, env_dir)
    return environment.id, environment_version.id


@pytest.fixture(scope="session")
def pytorch_drop_in_env():
    env_dir = os.path.join(PUBLIC_DROPIN_ENVS_PATH, "python3_pytorch")
    environment = dr.ExecutionEnvironment.create(
        name="python3_pytorch", programming_language="python"
    )
    environment_version = dr.ExecutionEnvironmentVersion.create(environment.id, env_dir)
    return environment.id, environment_version.id


@pytest.fixture(scope="session")
def onnx_drop_in_env():
    env_dir = os.path.join(PUBLIC_DROPIN_ENVS_PATH, "python3_onnx")
    environment = dr.ExecutionEnvironment.create(name="python3_onnx", programming_language="python")
    environment_version = dr.ExecutionEnvironmentVersion.create(environment.id, env_dir)
    return environment.id, environment_version.id


@pytest.fixture(scope="session")
def keras_drop_in_env():
    env_dir = os.path.join(PUBLIC_DROPIN_ENVS_PATH, "python3_keras")
    environment = dr.ExecutionEnvironment.create(
        name="python3_keras", programming_language="python"
    )
    environment_version = dr.ExecutionEnvironmentVersion.create(environment.id, env_dir)
    return environment.id, environment_version.id


@pytest.fixture(scope="session")
def pmml_drop_in_env():
    env_dir = os.path.join(PUBLIC_DROPIN_ENVS_PATH, "python3_pmml")
    environment = dr.ExecutionEnvironment.create(name="python3_pmml", programming_language="python")
    environment_version = dr.ExecutionEnvironmentVersion.create(environment.id, env_dir)
    return environment.id, environment_version.id


@pytest.fixture(scope="session")
def r_drop_in_env():
    env_dir = os.path.join(PUBLIC_DROPIN_ENVS_PATH, "r_lang")
    environment = dr.ExecutionEnvironment.create(name="r_drop_in", programming_language="r")
    environment_version = dr.ExecutionEnvironmentVersion.create(environment.id, env_dir)
    return environment.id, environment_version.id


@pytest.fixture(scope="session")
def julia_drop_in_env():
    env_dir = os.path.join(PUBLIC_DROPIN_ENVS_PATH, "julia_mlj")
    environment = dr.ExecutionEnvironment.create(name="julia_drop_in", programming_language="other")
    environment_version = dr.ExecutionEnvironmentVersion.create(environment.id, env_dir)
    return environment.id, environment_version.id


@pytest.fixture(scope="session")
def binary_testing_data():
    dataset = dr.Dataset.create_from_file(
        file_path=os.path.join(TESTS_DATA_PATH, "iris_binary_training.csv")
    )
    return dataset.id


@pytest.fixture(scope="session")
def binary_vizai_testing_data():
    dataset = dr.Dataset.create_from_file(
        file_path=os.path.join(TESTS_DATA_PATH, "cats_dogs_small_training.csv")
    )
    return dataset.id


@pytest.fixture(scope="session")
def regression_testing_data():
    dataset = dr.Dataset.create_from_file(
        file_path=os.path.join(TESTS_DATA_PATH, "juniors_3_year_stats_regression.csv")
    )
    return dataset.id


@pytest.fixture(scope="session")
def multiclass_testing_data():
    dataset = dr.Dataset.create_from_file(
        file_path=os.path.join(TESTS_DATA_PATH, "skyserver_sql2_27_2018_6_51_39_pm.csv")
    )
    return dataset.id


@pytest.fixture(scope="session")
def unstructured_testing_data():
    dataset = dr.Dataset.create_from_file(
        file_path=os.path.join(TESTS_DATA_PATH, "unstructured_data.txt")
    )
    return dataset.id
