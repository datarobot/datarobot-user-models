## Python pypmml template model

This model is intended to work with the [Python 3 pypmml Drop-In Environment](../../public_dropin_environments/python3_pypmml).
The supplied pmml file is a pypmml model trained on [iris_binary_training.csv](../../tests/testdata/iris_binary_training.csv)
with a Species as the target (classification), though any binary or regression model trained using the libraries
outlined in [Python 3 pypmml Drop-In Environment](../../public_dropin_environments/python3_pypmml) will work.

For this sample model, custom.py contains additional data pre-processing that the model itself lacks.

## Instructions
Create a new custom model with these files and use the Python Drop-In Environment with it

### To run locally using 'drum'
Paths are relative to `./datarobot-user-models/model_templates`:  
`drum score --code-dir ./python3_pypmml_inference --input ../tests/testdata/iris_binary_training.csv --positive-class-label Another-Iris --negative-class-label P_classIris_setosa`