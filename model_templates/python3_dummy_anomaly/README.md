## Python Dummy Anomaly Detection Inference Model Template

This anomaly detection model is a very simple example that yields fake results, regardless of the provided input dataset.
Every 10th row of any input dataset will yield probably of 75% of being an anomaly (0.75 anomaly score).
It works with any Python environment that has `pandas`.

## Instructions
Create a new custom model with this `custom.py` and use any Python Drop-In Environment with it.

### To run locally using 'drum'
Paths are relative to `./datarobot-user-models`:


`drum score --code-dir model_templates/python3_dummy_anomaly --target-type anomaly --input tests/testdata/juniors_3_year_stats_regression.csv`

Note: any input dataset will work for this model.
