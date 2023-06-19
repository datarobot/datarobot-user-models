## Python PMML Inference Model Template

This model is intended to work with the [Python 3 pypmml Drop-In Environment](../../public_dropin_environments/python3_pmml).

The supplied pmml file (iris_bin.pmml) is a pypmml model trained on [iris_binary_training.csv](../../tests/testdata/iris_binary_training.csv)
with a Species as the target (classification), though any binary or regression model trained using the libraries
outlined in [Python 3 pmml Drop-In Environment](../../public_dropin_environments/python3_pmml) will work.

## NOTE
You'll probably notice that the PMML file is not loaded explicitly in the custom.py file. Not to worry! In the drop-in environment we implicitly handle the score function for PMML so it is not needed, and therefore left out of the example. We handle certain other models types implicitly/automatically as well, like scikit-learn pickle files. 

## Instructions
Create a new custom model with these files and use the Python Drop-In Environment with it

### To run locally classification model using 'drum'
Paths are relative to `./datarobot-user-models`:  
`drum score --code-dir model_templates/python3_pmml --target-type binary --input tests/testdata/iris_binary_training.csv --positive-class-label Iris-setosa --negative-class-label Iris-versicolor`

### To run locally regression model using 'drum'
Replace `iris_bin.pmml` with `iris_reg.pmml` from the repo's `tests/fixtures/drop_in_model_atifacts` folder, which is a pypmml model trained on Iris dataset with a `Sepal Length` as the target (regression).  
Paths are relative to `./datarobot-user-models`:  
`drum score --code-dir model_templates/python3_pmml --target-type regression --input tests/testdata/iris_binary_training.csv`
