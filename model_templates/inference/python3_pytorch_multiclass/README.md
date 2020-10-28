## Python PyTorch Inference Model Template
This model is intended to work with the [Python 3 PyTorch Drop-In Environment](../../../public_dropin_environments/python3_pytorch/)
with additional dependencies.  This model includes a requirements file that will be installed
on top of the base environment specified if added to DataRobot.

In addition to the main model artifact, this model also has an artifact for the preprocessing pipeline

Because of these extra artifacts and preprocessing, custom.py uses the `load_model` and `transform` hooks
at predict time.

The supplied pth file is a PyTorch model trained on [skyserver_sql2_27_2018_6_51_39_pm.csv](../../../tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv)
with a `class` as the target (multiclass).

The model was trained using:
`drum fit -cd model_templates/training/python3_pytorch_multiclass/ --target-type multiclass --target class --input tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv --output model_templates/inference/python3_pytorch_multiclass/`

## Instructions
Create a new custom model with these files and use the Python Drop-In Environment with it

### To run locally using 'drum'
Paths are relative to `datarobot-user-models` root:  
`drum score --code-dir model_templates/inference/python3_pytorch_multiclass --input tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv --target-type multiclass --class-labels-file model_templates/inference/python3_pytorch_multiclass/class_labels.txt`