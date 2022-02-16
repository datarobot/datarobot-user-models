## Python ONNX Inference Anomaly Detection Model Template


This model is intended to work with the [Python 3 ONNX Drop-In Environment](../../public_dropin_environments/python3_onnx/).

The supplied .onnx file is the ONNX export of the [scikit-learn One Class SVM model](../python3_sklearn_anomaly) trained on [juniors_3_year_stats_regression.csv](../../tests/testdata/juniors_3_year_stats_regression.csv)
with no target. 

For this sample model, custom.py contains additional data pre-processing that the model itself lacks.

### To run locally using 'drum'
Paths are relative to `./datarobot-user-models`:   
`drum score --code-dir model_templates/python3_onnx_anomaly --target-type anomaly --input tests/testdata/juniors_3_year_stats_regression.csv`