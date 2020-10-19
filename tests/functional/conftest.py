import os
import uuid
import warnings

import datarobot as dr
import pytest
from dr_usertool.datarobot_user_database import DataRobotUserDatabase
from dr_usertool.utils import get_permissions

# fixtures dir
TESTS_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_FIXTURES_PATH = os.path.join(TESTS_ROOT_PATH, "fixtures")


BASE_TEMPLATE_ENV_DIR = "public_dropin_environments"
BASE_DEV_ENV_DIR = "dev_env"
BASE_DATASET_DIR = "tests/testdata"
ENDPOINT_URL = "http://localhost/api/v2"


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


def pytest_sessionstart(session):
    dr.Client(endpoint=ENDPOINT_URL, token=os.environ["DATAROBOT_API_TOKEN"])


@pytest.fixture(scope="session")
def java_drop_in_env():
    env_dir = os.path.join(BASE_TEMPLATE_ENV_DIR, "java_codegen")
    environment = dr.ExecutionEnvironment.create(name="java_drop_in")
    environment_version = dr.ExecutionEnvironmentVersion.create(environment.id, env_dir)
    return environment.id, environment_version.id


@pytest.fixture(scope="session")
def java_h2o_drop_in_env():
    env_dir = os.path.join(BASE_TEMPLATE_ENV_DIR, "java_h2o")
    environment = dr.ExecutionEnvironment.create(name="java_h2o_drop_in")
    environment_version = dr.ExecutionEnvironmentVersion.create(environment.id, env_dir)
    return environment.id, environment_version.id


@pytest.fixture(scope="session")
def sklearn_drop_in_env():
    env_dir = os.path.join(BASE_TEMPLATE_ENV_DIR, "python3_sklearn")
    environment = dr.ExecutionEnvironment.create(name="python3_sklearn")
    environment_version = dr.ExecutionEnvironmentVersion.create(environment.id, env_dir)
    return environment.id, environment_version.id


@pytest.fixture(scope="session")
def xgboost_drop_in_env():
    env_dir = os.path.join(BASE_TEMPLATE_ENV_DIR, "python3_xgboost")
    environment = dr.ExecutionEnvironment.create(name="python3_xgboost")
    environment_version = dr.ExecutionEnvironmentVersion.create(environment.id, env_dir)
    return environment.id, environment_version.id


@pytest.fixture(scope="session")
def pytorch_drop_in_env():
    env_dir = os.path.join(BASE_TEMPLATE_ENV_DIR, "python3_pytorch")
    environment = dr.ExecutionEnvironment.create(name="python3_pytorch")
    environment_version = dr.ExecutionEnvironmentVersion.create(environment.id, env_dir)
    return environment.id, environment_version.id


@pytest.fixture(scope="session")
def keras_drop_in_env():
    env_dir = os.path.join(BASE_TEMPLATE_ENV_DIR, "python3_keras")
    environment = dr.ExecutionEnvironment.create(name="python3_keras")
    environment_version = dr.ExecutionEnvironmentVersion.create(environment.id, env_dir)
    return environment.id, environment_version.id


@pytest.fixture(scope="session")
def pmml_drop_in_env():
    env_dir = os.path.join(BASE_TEMPLATE_ENV_DIR, "python3_pmml")
    environment = dr.ExecutionEnvironment.create(name="python3_pmml")
    environment_version = dr.ExecutionEnvironmentVersion.create(environment.id, env_dir)
    return environment.id, environment_version.id


@pytest.fixture(scope="session")
def r_drop_in_env():
    env_dir = os.path.join(BASE_TEMPLATE_ENV_DIR, "r_lang")
    environment = dr.ExecutionEnvironment.create(name="r_drop_in")
    environment_version = dr.ExecutionEnvironmentVersion.create(environment.id, env_dir)
    return environment.id, environment_version.id


@pytest.fixture(scope="session")
def binary_testing_data():
    dataset = dr.Dataset.create_from_file(
        file_path=os.path.join(BASE_DATASET_DIR, "iris_binary_training.csv")
    )
    return dataset.id


@pytest.fixture(scope="session")
def binary_vizai_testing_data():
    dataset = dr.Dataset.create_from_file(
        file_path=os.path.join(BASE_DATASET_DIR, "cats_dogs_small_training.csv")
    )
    return dataset.id


@pytest.fixture(scope="session")
def regression_testing_data():
    dataset = dr.Dataset.create_from_file(
        file_path=os.path.join(BASE_DATASET_DIR, "boston_housing.csv")
    )
    return dataset.id


@pytest.fixture(scope="session")
def multiclass_testing_data():
    dataset = dr.Dataset.create_from_file(
        file_path=os.path.join(BASE_DATASET_DIR, "Skyserver_SQL2_27_2018 6_51_39 PM.csv")
    )
    return dataset.id
