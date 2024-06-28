import os


def abspath(rltv_path):
    return os.path.join(os.path.dirname(__file__), rltv_path)


MLPIPER_JAR_FILEPATH = abspath("../../mlpiper-java/target/mlpiper.jar")

COMPONENTS_ROOT = abspath("resources/steps")
GENERIC_COMPONENTS_ROOT = os.path.join(COMPONENTS_ROOT, "Generic")
PYTHON_COMPONENTS_PATH = os.path.join(GENERIC_COMPONENTS_ROOT, "Python")
JAVA_COMPONENTS_PATH = os.path.join(GENERIC_COMPONENTS_ROOT, "Java")
R_COMPONENTS_PATH = os.path.join(GENERIC_COMPONENTS_ROOT, "R")
PYSPARK_COMPONENTS_PATH = os.path.join(COMPONENTS_ROOT, "PySpark")
REST_COMPONENTS_PATH = os.path.join(COMPONENTS_ROOT, "RestModelServing")
SAGEMAKER_COMPONENTS_PATH = os.path.join(COMPONENTS_ROOT, "SageMaker")
