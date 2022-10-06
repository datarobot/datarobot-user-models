node('multi-executor && ubuntu:focal'){
  checkout scm
  stage ('Checkout') {
      checkout scm
  }
  dir('jenkins_artifacts'){
      unstash 'drum_wheel'
  }
  sh "ls -la jenkins_artifacts"
  sh "echo $PIPELINE_CONTROLLER"
  sh 'bash jenkins/test_integration.sh'
}