def build() {
    stage('Build test') {
        sh """
            echo "this is build stage"
            echo $WORKSPACE
            touch $WORKSPACE/yakoff_file
            ls $WORKSPACE
        """
    }
}

def test_python_sklearn() {
    stage('Run test_python_sklearn') {
        sh """
            echo "this is test_python_sklearn"
            echo $WORKSPACE
            ls $WORKSPACE
            touch $WORKSPACE/yakoff_file2
        """
    }
}

def test_python_xgboost() {
    stage('Run test_python_xgboost') {
        sh """
            echo "this is test_python_xgboost"
            echo $WORKSPACE
            ls $WORKSPACE
        """
    }
}

return this