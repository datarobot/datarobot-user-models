node('multi-executor && ubuntu:focal'){
    stage('build_drum') {
        checkout scm
        sh """
            echo $WORKSPACE
            source "tools/image-build-utils.sh"
            build_drum
            ls -la custom_model_runner/dist/*
        """
        dir ('custom_model_runner/dist') {
            stash includes: '*', name: 'drum_wheel'
        }
    }
}
