## Python Sklearn Inference Anomaly Detection Model Template


This model is intended to work with the [Python 3 Scikit-Learn Drop-In Environment](../../public_dropin_environments/python3_sklearn/).
The supplied pkl file is a scikit-learn One Class SVM model trained on [juniors_3_year_stats_regression.csv](../../tests/testdata/juniors_3_year_stats_regression.csv)
with no target. 

For fitting the model, use the following [training model](../../task_templates/2_estimators/7_python_anomaly)

### To run locally using 'drum'
Paths are relative to `./datarobot-user-models`:   
`drum score --code-dir model_templates/python3_sklearn_anomaly --target-type anomaly --input tests/testdata/juniors_3_year_stats_regression.csv`
