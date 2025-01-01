#!/usr/bin/env bash

echo "== Preparing to test =="
apt-get update && apt-get install -y curl

MLOPS_VERSION="9.2.8"
MLOPS_AGENT_VERSION=${MLOPS_VERSION}
MLOPS_AGENT_JAR_DIR="/tmp/jars"
REMOTE_FILE="https://repo1.maven.org/maven2/com/datarobot/mlops-agent/${MLOPS_AGENT_VERSION}/mlops-agent-${MLOPS_AGENT_VERSION}.jar"

pip install datarobot-mlops==${MLOPS_VERSION}

mkdir -p "${MLOPS_AGENT_JAR_DIR}"
curl --output "${MLOPS_AGENT_JAR_DIR}"/mlops-agent-${MLOPS_AGENT_VERSION}.jar ${REMOTE_FILE}
export MLOPS_MONITORING_AGENT_JAR_PATH=${MLOPS_AGENT_JAR_DIR}/mlops-agent-${MLOPS_AGENT_VERSION}.jar

echo "Installing requirements for all the tests: requirements_test.txt"
pip install -r requirements_test.txt

echo "== Uninstall drum =="
pip uninstall -y datarobot-drum

cd custom_model_runner
echo "== Install drum from source =="
pip install .
cd -

# pytest tests/functional/test_mlops_monitoring.py -k "not test_drum_unstructured_model_embedded_monitoring_in_sklearn_env" -n 1
pytest tests/functional/test_mlops_monitoring.py