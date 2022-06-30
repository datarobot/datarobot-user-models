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
            jenkins/test3_mlpiper_custom_models_without_container.sh
        '''.stripIndent(),
        pythonVersion: '3',
        venvName: "datarobot-user-models"
    ])
  }
}