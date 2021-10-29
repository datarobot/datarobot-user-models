## Python Keras Joblib Estimator Template
This model is intended to work with the [Python 3 Keras Drop-In Environment](../../../public_dropin_environments/python3_keras/).

For this sample estimator, custom.py contains additional data pre-processing that the model itself lacks.  In this case,
preprocessing consists of selecting only numerical columns and performing median imputation followed by scaling. Note
that any columns that are entirely NaN will be dropped by this preprocessing pipeline.

## Instructions
Create a new custom model with these files and use the Python Drop-In Environment with it

### To run locally using 'drum'
Paths are relative to `datarobot-user-models` root:  
`drum fit --code-dir task_templates/3_pipelines/python3_keras_joblib --input tests/testdata/juniors_3_year_stats_regression.csv --target-type regression --target "Grade 2014"`  
Check that `artifact.joblib` file was created.
