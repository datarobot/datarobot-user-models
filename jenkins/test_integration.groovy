node('multi-executor && ubuntu:focal'){
    checkout scm

    /*
        Integration tests require java predictor to be built and java is required to run some checks.
        So the easiest and fastest way is to run these tests in the drum builder container, which has python/java/maven.
    */
    docker.image('datarobot/drum-builder').inside() {
        stage('test_integration') {
            try {
                sh"""#!/bin/bash
                set -exuo pipefail
                cd custom_model_runner
                make
                cd -
                python3 -m venv /tmp/venv
                . /tmp/venv/bin/activate
                pip3 install -U pip
                pip3 install -r requirements_test_unit.txt
                pip3 install -e custom_model_runner/
                pytest tests/integration --junit-xml="unit-test-report.xml"
                """
            } finally {
                junit allowEmptyResults: true, testResults: '**/unit-test-report.xml'
            }
        }
    }
}
