node('multi-executor && ubuntu:focal'){
  checkout scm
  stage ('test_integration_all_in_one_bare_metal') {
    checkout scm

    dir('jenkins_artifacts'){
        unstash 'drum_wheel'
    }
    withQuantum([
        bash: '''\
            set -exuo pipefail
            ls -la jenkins_artifacts
            jenkins/test_integration_general.sh "$(realpath "$(find jenkins_artifacts/datarobot_drum*.whl)")"
        '''.stripIndent(),
        pythonVersion: '3',
        venvName: "datarobot-user-models"
    ])
  }
}