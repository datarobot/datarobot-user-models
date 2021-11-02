## Python Unsupervised Anomaly Detection Training Model Template

This model is intended to work with the [Python 3 Scikit-Learn Drop-In Environment](../../../public_dropin_environments/python3_sklearn/).

Note that this template includes code to apply calibration to your anomaly detector's predictions. 
Calibration is a way to coerce your raw predictions into values that can be interpreted as probabilities;
calibrated predictions will be floats in the range [0.0, 1.0]. 

Using calibration is optional, and if you choose not to calibrate your model directly you may opt to allow 
DataRobot to calibrate your model predictions for you. A calibrated model has the advantages that:
1. ROC curve insights will be available for external scoring data*
2. Model comparison is likely to be more trustworthy, since all predictions will be on the same scale 

The classes defined in `anomaly_helpers.py` demonstrate one approach to including a calibration step 
with your estimator. Note that the given calibration method is very rudimentary, and calibration
will likely not be necessary for models that already have a `predict_proba()` method.

\* External scoring data for anomaly detection is labeled with actual 0-1 indicators of true anomalies. This can be uploaded in the predict tab, after which all leadboard models may be scored against these labels. Any model scored on an external dataset in this manner will have access to lift chart and, if calibrated, ROC curve insights.

## Instructions
Create a new custom model with these files and use the Python Drop-In Environment with it

### To run locally using 'drum'
Paths are relative to `datarobot-user-models` root:  
`drum fit --code-dir task_templates/3_pipelines/python3_anomaly_detection --input tests/testdata/juniors_3_year_stats_regression.csv --target-type anomaly`
