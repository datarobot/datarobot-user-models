node('multi-executor && ubuntu:focal'){
    checkout scm

    docker.image('python:3.9').inside() {
        stage('black_check') {
            sh"""#!/bin/bash
            set -exuo pipefail
            python -m venv /tmp/venv
            . /tmp/venv/bin/activate
            pip install -U pip
            pip install -r requirements_lint.txt
            black --check --diff .
            """
        }
    }
}