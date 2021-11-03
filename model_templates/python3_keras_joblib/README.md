## Python Keras Joblib Inference Model Template

This model is intended to work with the [Python 3 Keras Drop-In Environment](../../public_dropin_environments/python3_keras/).
The supplied h5 file is a keras + tensorflow model trained on [juniors_3_year_stats_regression.csv](../../tests/testdata/juniors_3_year_stats_regression.csv)
with a `Grade 2014` as the target (regression), though any binary or regression model trained using the libraries
outlined in [Python 3 Keras Drop-In Environment](../../public_dropin_environments/python3_keras/) will work.

For this sample model, custom.py contains additional data pre-processing that the model itself lacks.

For fitting the model, use the following [training model](../../task_templates/3_pipelines/python3_keras_joblib)

## Instructions
Create a new custom model with these files and use the Python Drop-In Environment with it

### To run locally using 'drum'
Paths are relative to `./datarobot-user-models`:  
`drum score --code-dir model_templates/python3_keras_joblib --target-type regression --input tests/testdata/juniors_3_year_stats_regression.csv`
