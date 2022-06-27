def steps
node('multi-executor && ubuntu:focal'){
    stage ('Checkout') {
        cleanWs()
        checkout scm
        steps = load 'jenkins/steps.groovy'
    }

    steps.build()
    steps.test_python_sklearn()
    steps.test_python_xgboost()

}