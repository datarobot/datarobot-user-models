## Python Keras Vizai Training Model Template

This model is intended to work with the [Python 3 Keras Drop-In Environment](../../public_dropin_environments/python3_keras/).

## Instructions
Create a new custom model with these files and use the Python Drop-In Environment with it

### To run locally using 'drum'
Paths are relative to `./datarobot-user-models/model_templates`:  
`drum fit --code-dir ./python3_keras_vizai_training_joblib --input ../tests/testdata/cats_dogs_small_training.csv --target class --positive-class-label cats --negative-class-label dogs --output ./ --skip-predict`  
Check that `artifact.joblib` file was created.