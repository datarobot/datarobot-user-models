## Python pypmml template model

These models are intended to work with the [Python 3 pypmml Drop-In Environment](../../public_dropin_environments/python3_pypmml).

The supplied pmml file (iris_bin.pmml) is a pypmml model trained on [iris_binary_training.csv](../../tests/testdata/iris_binary_training.csv) 
with a Species as the target (classification), though any binary or regression model trained using the libraries
outlined in [Python 3 pypmml Drop-In Environment](../../public_dropin_environments/python3_pypmml) will work.

The supplied pmml file (iris_reg.pmml) is a pypmml model trained on Iris dataset 
with a 'Sepal Length' as the target (regression).

For these sample models, custom.py contains additional data pre-processing that the model itself lacks.

## Instructions
Create a new custom model with these files and use the Python Drop-In Environment with it

### To run locally classification model using 'drum'
Paths are relative to `./datarobot-user-models/model_templates`:  
`drum score --code-dir ./python3_pypmml_inference --input ../tests/testdata/iris_binary_training.csv --positive-class-label Iris-setosa --negative-class-label Iris-versicolor`

### To run locally regression model using 'drum'
Paths are relative to `./datarobot-user-models/model_templates`:  
`drum score --code-dir ./python3_pypmml_inference --input ../tests/testdata/iris_binary_training.csv`