node('release-dev && memory-intense'){
  stage ('test_drop_in_envs') {
    checkout scm

    dir('jenkins_artifacts'){
        unstash 'drum_wheel'
    }

    sh"""#!/bin/bash
    echo "Install Python 3.8, because Quantum's Py3.7 is too old for some packages "
    sudo apt-get update
    sudo apt-get install -y python3.8 python3.8-dev python3.8-distutils python3.8-venv
    python3.8 -m venv /tmp/venv_py_3_8
    """

    checkoutDataRobot()
    sh '''
        set -exuo pipefail
        pushd DataRobot
        make update_env
        export LRS_POLL_DELAY_SECONDS=1
        export LRS_RETRIES=120
        export CUSTOM_MODEL_PREDICT_MEM_LIMIT=2147483648
        export CUSTOM_MODEL_PREDICT_MEM_REQUEST=2147483648
        export EXECUTION_ENVIRONMENT_LIMIT=20
        export RESTRICT_EXECUTION_ENVIRONMENT_CREATION=false
        export CLIENT_VERSION=\$(git rev-parse --short HEAD)
        ./start.sh --kubernetes-k3d --kubernetes-validate
        popd
    '''.stripIndent()

    createInitialAdminUser()

    try {
        sh"""#!/bin/bash
        set -exuo pipefail
        . /tmp/venv_py_3_8/bin/activate
        ls -la jenkins_artifacts
        ./jenkins/test_drop_in_envs.sh
        """
    } finally {
      junit allowEmptyResults: true, testResults: '**/results*.xml'
    }
  }
}
