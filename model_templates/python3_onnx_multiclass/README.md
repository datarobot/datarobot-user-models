## Python ONNX Inference Multiclass Model Template


This model is intended to work with the [Python 3 ONNX Drop-In Environment](../../public_dropin_environments/python3_onnx/).

In addition to the main model artifact, this model also has an artifact for the preprocessing pipeline. Because of these extra artifacts and preprocessing, custom.py uses the `load_model` and `transform` hooks at predict time.

The supplied .onnx file is the ONNX export of a [PyTorch model](../python3_pytorch_multiclass) trained on [skyserver_sql2_27_2018_6_51_39_pm.csv](../../tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv)
with `class` as the target (multiclass).


### To run locally using 'drum'
Paths are relative to `datarobot-user-models` root:  
`drum score --code-dir model_templates/python3_onnx_multiclass --input tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv --target-type multiclass --class-labels-file model_templates/python3_pytorch_multiclass/class_labels.txt`