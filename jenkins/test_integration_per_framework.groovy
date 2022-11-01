node('multi-executor && ubuntu:focal'){
  checkout scm
  stage ('Checkout') {
      checkout scm
  }
  dir('jenkins_artifacts'){
      unstash 'drum_wheel'
  }
  try {
    sh "ls -la jenkins_artifacts"
    sh "echo $FRAMEWORK"
    sh 'bash jenkins/test_integration_per_framework.sh $FRAMEWORK'
  } finally {
    junit allowEmptyResults: true, testResults: '**/results*.xml'
  }
}
