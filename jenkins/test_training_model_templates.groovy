node('release-dev && memory-intense'){
  stage ('test_training_model_templates') {
    checkout scm

    dir('jenkins_artifacts'){
        unstash 'drum_wheel'
    }

    checkoutDataRobot()

    sh '''
        set -exuo pipefail
        pushd DataRobot
        make update_env
        export LRS_POLL_DELAY_SECONDS=1
        export LRS_RETRIES=120
        export CUSTOM_MODEL_PREDICT_MEM_LIMIT=2147483648
        export CUSTOM_MODEL_PREDICT_MEM_REQUEST=2147483648
        export EXECUTION_ENVIRONMENT_LIMIT=20
        export RESTRICT_EXECUTION_ENVIRONMENT_CREATION=false
        export CLIENT_VERSION=\$(git rev-parse --short HEAD)
        ./start.sh --kubernetes-k3d --kubernetes-validate
        popd
    '''.stripIndent()

    createInitialAdminUser()

    try {
      withQuantum([
          bash: '''\
              set -exuo pipefail
              ls -la jenkins_artifacts
              ./jenkins/test_training_model_templates.sh
          '''.stripIndent(),
          pythonVersion: '3',
          venvName: "datarobot-user-models"
      ])
    } finally {
      junit allowEmptyResults: true, testResults: '**/results*.xml'
    }
  }
}
