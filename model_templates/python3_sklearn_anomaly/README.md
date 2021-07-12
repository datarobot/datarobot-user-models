## Python Sklearn Inference Anomaly Detection Model Template


This model is intended to work with the [Python 3 Scikit-Learn Drop-In Environment](../../../public_dropin_environments/python3_sklearn/).
The supplied pkl file is a scikit-learn model trained on [boston_housing.csv](../../../tests/testdata/boston_housing.csv)
with no target. 

### To run locally using 'drum'
Paths are relative to `./datarobot-user-models`:   
`drum score --code-dir model_templates/inference/python3_sklearn_anomaly --target-type anomaly --input tests/testdata/boston_housing.csv`
