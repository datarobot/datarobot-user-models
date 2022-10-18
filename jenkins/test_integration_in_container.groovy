node('multi-executor && ubuntu:focal'){
  checkout scm
  stage ('test_integration_in_container') {
    checkout scm

    dir('jenkins_artifacts'){
        unstash 'drum_wheel'
    }
    withQuantum([
        bash: '''\
            set -exuo pipefail
            ls -la jenkins_artifacts
            bash jenkins/test_integration_in_container.sh
        '''.stripIndent(),
        pythonVersion: '3',
        venvName: "datarobot-user-models"
    ])
  }
}