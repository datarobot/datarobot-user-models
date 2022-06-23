node('multi-executor && ubuntu:focal'){
    echo "Build DRUM"
    stage('Checkout') {
        cleanWs()
        checkout scm
    }

    stage('Build DRUM') {
        echo "I'm build stage"
        sh """
            echo "I'm build stage from shell"
            echo $WORKSPACE

            source "tools/image-build-utils.sh"
            build_drum
            ls -la custom_model_runner/dist/*
        """
    }
    dir ('custom_model_runner/dist') {
        stash includes: '*', name: 'drum_wheel'
    }
}