## Python Template Model

This model is intended to work with the [Python 3 XGBoost Drop-In Environment](../../public_dropin_environments/python3_xgboost/).
The supplied pkl file is an xgboost model trained on [boston_housing.csv](../../tests/testdata/boston_housing.csv)
with a MEDV as the target (regression), though any binary or regression model trained using the libraries
outlined in [Python 3 XGBoost Drop-In Environment](../../public_dropin_environments/python3_xgboost) will work.

For this sample model, custom.py contains additional data pre-processing that the model itself lacks.

## Instructions
Create a new custom model with these files and use the Python Drop-In Environment with it

### To run locally using 'drum'
Paths are relative to `./datarobot-user-models/model_templates`:  
`drum fit --code-dir ./python3_xgboost_training --input ../tests/testdata/boston_housing.csv --target MEDV  --output ./ --skip-predict`