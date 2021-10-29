## Python Pytorch Multiclass Training Model Template

This model is intended to work with the [Python 3 PyTorch Drop-In Environment](../../../public_dropin_environments/python3_pytorch/)
with additional dependencies.  This model includes a requirements file that will be installed
on top of the base environment specified if added to DataRobot.

In addition to the main model artifact, this model also produces an artifact for the preprocessing pipeline as well as a class_labels.txt
to make scoring with drum easier.

Because of these extra artifacts and preprocessing, custom.py uses the `load_model` and `transform` hooks
at predict time.

## Instructions
Create a new custom model with these files and use the Python Drop-In Environment with it

### To run locally using 'drum'
Paths are relative to `datarobot-user-models` root:  
`drum fit --code-dir task_templates/3_pipelines/python3_pytorch_multiclass --input tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv --target-type multiclass --target class`  
