node('multi-executor && ubuntu:focal'){
    checkout scm

    docker.image('python:3.8').inside() {
        stage('test_unit')
            try {
                sh"""#!/bin/bash
                set -exuo pipefail
                python -m venv /tmp/venv
                . /tmp/venv/bin/activate
                pip install -U pip
                pip install -r requirements_test_unit.txt
                pip install -e custom_model_runner/
                pytest tests/unit --junit-xml="unit-test-report.xml"
                """
            } finally {
              junit allowEmptyResults: true, testResults: '**/unit-test-report.xml'
            }
    }
}
