node('multi-executor && ubuntu:focal'){
  checkout scm
  stage ('Checkout') {
      checkout scm
  }
  dir('jenkins_artifacts'){
      unstash 'drum_wheel'
  }
  sh "ls -la jenkins_artifacts"
  withQuantum([
      bash: '''\
           set -exuo pipefail
           bash jenkins/test_integration_in_container.sh
      '''.stripIndent(),
      pythonVersion: '3',
      venvName: "datarobot-user-models"
  ])
}