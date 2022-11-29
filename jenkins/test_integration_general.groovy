node('multi-executor && ubuntu:focal'){
  stage ('test_integration_all_in_one_bare_metal') {
    checkout scm

    dir('jenkins_artifacts'){
        unstash 'drum_wheel'
    }
    try {
      withQuantum([
          bash: '''\
              set -exuo pipefail
              ls -la jenkins_artifacts
              jenkins/test_integration_general.sh
          '''.stripIndent(),
          pythonVersion: '3',
          venvName: "datarobot-user-models"
      ])
    } finally {
      junit allowEmptyResults: true, testResults: '**/results*.xml'
    }
  }
}
