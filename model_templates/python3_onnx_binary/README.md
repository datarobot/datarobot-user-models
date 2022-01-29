## Python ONNX Inference Binary Classification Model Template


This model is intended to work with the [Python 3 ONNX Drop-In Environment](../../public_dropin_environments/python3_onnx/).

The supplied .onnx file is the ONNX export of a PyTorch model trained on [iris_binary_training.csv](../../tests/testdata/iris_binary_training.csv)
with `Species` as the target (binary classification). The model outputs softmax-ed values for the positive and negative class probabilities.

For this sample model, custom.py contains additional data pre-processing that the model itself lacks.


### To run locally using 'drum'
Paths are relative to `./datarobot-user-models`:   
`drum score --code-dir model_templates/python3_onnx_binary --target-type binary --input tests/testdata/iris_binary_training.csv`