node('multi-executor && ubuntu:focal'){
  stage ('test_functional_all_in_one_bare_metal') {
    checkout scm

    dir('jenkins_artifacts'){
        unstash 'drum_wheel'
    }
    try {
        sh"""#!/bin/bash
        set -exuo pipefail
        ls -la jenkins_artifacts
        jenkins/test_functional_general.sh
        """
    } finally {
      junit allowEmptyResults: true, testResults: '**/results*.xml'
    }
  }
}
