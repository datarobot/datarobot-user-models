## Python Unsupervised Anomaly Detection Training Model Template

This model is intended to work with the [Python 3 Scikit-Learn Drop-In Environment](../../../public_dropin_environments/python3_sklearn/).

Custom training anomaly detection models have access to the same capabilities and insights as DR anomaly detection, with certain exceptions depending on whether your model is calibrated.
This template provides a sample uncalibrated model. For more details on calibration in the context of anomaly detection, see the  [calibrated anomaly detection template](../python3_calibrated_anomaly_detection/).

Note that custom training anomaly models will perform a variant on permutation feature impact. Baseline predictions are taken from the unaltered data, then the impact of a given feature is derived by permuting that column randomly and measuring the distance of the model's predictions with this permutation from the baseline.


## Instructions
Create a new custom model with these files and use the Python Drop-In Environment with it

### To run locally using 'drum'
Paths are relative to `datarobot-user-models` root:  
`drum fit --code-dir task_templates/pipelines/python3_anomaly_detection --input tests/testdata/juniors_3_year_stats_regression.csv --target-type anomaly`
