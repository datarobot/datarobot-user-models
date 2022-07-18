node('multi-executor && ubuntu:focal'){
  checkout scm
  stage ('Checkout') {
      checkout scm
  }
  dir('jenkins_artifacts'){
      unstash 'drum_wheel'
  }
  sh "ls -la jenkins_artifacts"
  sh "echo $FRAMEWORK"
  sh 'bash jenkins/run_integration_tests_per_framework.sh $FRAMEWORK'
}