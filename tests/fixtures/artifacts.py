import os
import glob
import shutil
import subprocess
from tempfile import mkdtemp

# fixtures dir
TESTS_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TESTS_FIXTURES_PATH = os.path.join(TESTS_ROOT_PATH, "fixtures")
TESTS_DATASET_DIR = os.path.join(TESTS_ROOT_PATH, "testdata")
SOURCE_ARTIFACTS_PATH = os.path.join(TESTS_FIXTURES_PATH, "drop_in_model_artifacts")


def generate_artifacts_dir():
    test_artifacts_dir = mkdtemp(prefix="custom_model_tests_artifacts", dir="/tmp")

    # copy files required to generate model artifacts
    for filename in glob.glob(r"{}/*.csv".format(TESTS_DATASET_DIR)):
        shutil.copy2(filename, test_artifacts_dir)

    for file_wildcard in ["*.ipynb", "PyTorch.*", "*.rds", "*.jar", "*.pmml"]:
        for filename in glob.glob(r"{}/{}".format(SOURCE_ARTIFACTS_PATH, file_wildcard)):
            shutil.copy2(filename, test_artifacts_dir)

    cur_dir = os.getcwd()
    try:
        os.chdir(test_artifacts_dir)
        for filename in glob.glob(r"{}/*ipynb".format(test_artifacts_dir)):
            subprocess.call(["jupyter", "nbconvert", "--execute", filename])
        subprocess.call(["python3", "PyTorch.py"])
    finally:
        os.chdir(cur_dir)
    return test_artifacts_dir
