## Python Keras Joblib Inference Model Template

This model is intended to work with the [Python 3 Keras Drop-In Environment](../../public_dropin_environments/python3_keras/).
The supplied _joblib_ file is a keras + tensorflow model trained on image dataset - [cats_dogs_small_training.csv](../../tests/testdata/cats_dogs_small_training.csv)
with _"class"_ as the target (binary classification), though any binary image model trained using the libraries
outlined in [Python 3 Keras Drop-In Environment](../../public_dropin_environments/python3_keras/) will work.

## Instructions
Create a new custom model with these files and use the Python Drop-In Environment with it

### To run locally using 'drum'
Paths are relative to `./datarobot-user-models/model_templates`:  
`drum score --code-dir ./inference/python3_keras_vizai_joblib --input ../tests/testdata/cats_dogs_small_training.csv --verbose`
