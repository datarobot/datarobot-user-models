
#MON_SETTINGS="spooler_type=filesystem;directory=/tmp/tt;max_files=9;file_max_size=1024000"

#export MONITOR_SETTINGS=$MON_SETTINGS
#export MODEL_IDD=7777
#export DEPLOYMENT_ID=8888
#CODE_DIR="../model_templates/inference/python3_sklearn/"
#INPUT_FILE="../tests/testdata/boston_housing_inference.csv"
#LABELS=" "

CODE_DIR="../tools/r2d2"
INPUT_FILE="../tests/testdata/iris_binary_training.csv"
LABELS=" --positive-class-label Iris-setosa --negative-class-label Iris-versicolor "

 

set -x
env PYTHONPATH=./ \
        ./bin/drum automem \
        --code-dir $CODE_DIR\
        --input $INPUT_FILE \
        --target-type regression \
        --min-mem 500 \
        --max-mem 1600 \
        --step 256 \
        --verbose

