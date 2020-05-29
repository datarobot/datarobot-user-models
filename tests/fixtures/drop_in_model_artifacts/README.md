## Drop-In Environment
These model artifacts are meant to be used with the drop-in template environments in this [repo](https://github.com/datarobot/custom-model-templates/tree/master/custom_environment_templates)

## Artifact Details
For python, you will need to install the libraries from [here](https://github.com/datarobot/custom-model-templates/blob/master/custom_environment_templates/python_3/py3_drop_in/dr_requirements.txt)

For R you will need an environment with the libraries installed in this [Dockerfile](https://github.com/datarobot/custom-model-templates/blob/master/custom_environment_templates/r_environment/r_drop_in_environment/Dockerfile)

The Java model artifacts were made using codegen.

### Datasets used 

 - Binary classification [iris_binary_training.csv](../../testdata/iris_binary_training.csv)
   - This dataset does not have enough rows for DR. To get the Java Scoring Code models, I just copied the dataset.
 - Regression [boston_housing.csv](../../testdata/boston_housing.csv)
 
### Generating new artifacts

 - XGBoost, SKLearn, Keras: run the associated notebooks in this directory
 - PyTorch: run PyTorch.py
   - This is a separate script because the custom model needs to include the tourch.nn class for the model as well as the model artifact
 - R: run Rmodel.R
 - Java:
   - create projects with the desired dataset (double the iris dataset)
   - train a model that generates scoring code
   - download scoring code
   - rename jar to `java_bin.jar` or `java_reg.jar`
