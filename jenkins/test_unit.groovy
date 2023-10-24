node('multi-executor && ubuntu:focal'){
    checkout scm

    docker.image('python:3.8').inside() {
        stage('test_unit') {
            sh"""#!/bin/bash
            set -exuo pipefail
            python -m venv /tmp/venv
            . /tmp/venv/bin/activate
            pip install -U pip
            pip install -r requirements_test.txt
            pytest tests/unit
            """
        }
    }
}