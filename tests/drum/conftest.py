import os
import pytest
import shutil
import sys


# fixtures dir
TESTS_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_FIXTURES_PATH = os.path.join(TESTS_ROOT_PATH, "fixtures")

sys.path.append(TESTS_FIXTURES_PATH)
from artifacts import generate_artifacts_dir

test_artifacts_dir = None


@pytest.fixture(scope="session")
def get_artifacts_dir():
    return test_artifacts_dir


def pytest_sessionstart(session):
    global test_artifacts_dir
    test_artifacts_dir = generate_artifacts_dir()


def pytest_sessionfinish(session):
    global test_artifacts_dir
    shutil.rmtree(test_artifacts_dir)
