node('multi-executor && ubuntu:focal'){
    stage('build_drum') {
        checkout scm
        withQuantum([
            bash: '''\
                set -exuo pipefail
                pip install -U pip
                pip install --only-binary=:all: -r requirements_lint.txt
                black --check .
            '''.stripIndent(),
            pythonVersion: '3',
            venvName: "datarobot-user-models"
        ])
    }
}