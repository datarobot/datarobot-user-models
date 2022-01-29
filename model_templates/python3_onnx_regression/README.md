## Python ONNX Inference Regression Model Template


This model is intended to work with the [Python 3 ONNX Drop-In Environment](../../public_dropin_environments/python3_onnx/).

The supplied .onnx file is the ONNX export of the [PyTorch model](../python3_pytorch) trained on [juniors_3_year_stats_regression.csv](../../tests/testdata/juniors_3_year_stats_regression.csv)
with `Grade 2014` as the target (regression).

For this sample model, custom.py contains additional data pre-processing that the model itself lacks.


### To run locally using 'drum'
Paths are relative to `./datarobot-user-models`:   
`drum score --code-dir model_templates/python3_onnx_regression --target-type regression --input tests/testdata/juniors_3_year_stats_regression.csv`
