## Python XGBoost Training Model Template

This model is intended to work with the [Python 3 XGBoost Drop-In Environment](../../../public_dropin_environments/python3_xgboost/).
The supplied pkl file is an xgboost model trained on [boston_housing.csv](../../../tests/testdata/boston_housing.csv)
with a MEDV as the target (regression), though any binary or regression model trained using the libraries
outlined in [Python 3 XGBoost Drop-In Environment](../../../public_dropin_environments/python3_xgboost) will work.

For this sample model, custom.py contains additional data pre-processing that the model itself lacks. In this case,
preprocessing consists of selecting only numerical columns and performing median imputation followed by scaling. Note
that any columns that are entirely NaN will be dropped by this preprocessing pipeline.

## Instructions
Create a new custom model with these files and use the Python Drop-In Environment with it

### To run locally using 'drum'
Paths are relative to `datarobot-user-models` root:  
`drum fit --code-dir model_templates/training/python3_xgboost --input tests/testdata/boston_housing.csv --target-type regression --target MEDV`  
Check that `artifact.pkl` file was created.
