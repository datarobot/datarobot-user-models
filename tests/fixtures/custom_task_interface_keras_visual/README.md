## Python Keras Visual AI Training Model Template

This model is intended to work with the [Python 3 Keras Drop-In Environment](../../public_dropin_environments/python3_keras/).

## Instructions
Create a new custom model with these files and use the Python Drop-In Environment with it

### To run locally using 'drum'
Paths are relative to `datarobot-user-models` root:  
`drum fit --code-dir ./task_templates/3_pipelines/15_python3_keras_vizai_joblib/ --input ./tests/testdata/cats_dogs_small_training.csv --target-type binary --target class --positive-class-label cats --negative-class-label dogs --output ./ --skip-predict --verbose`
Check that `artifact.joblib` file was created.