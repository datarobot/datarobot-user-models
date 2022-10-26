node('multi-executor && ubuntu:focal'){
    stage ('test_integration_per_framework $FRAMEWORK') {
        checkout scm
        dir('jenkins_artifacts'){
            unstash 'drum_wheel'
        }

        sh '''
            set -exuo pipefail
            ls -la jenkins_artifacts
            echo $FRAMEWORK
            jenkins/test_integration_per_framework.sh "$(realpath "$(find jenkins_artifacts/datarobot_drum*.whl)")" $FRAMEWORK
        '''
    }
}