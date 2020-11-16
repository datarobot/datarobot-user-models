## Sklearn Transform Template

This is a python transformer template with preproccessing that:
- Handles categorical variables
- Handles numeric variables
- Log-transforms the target variable

This transformer is intended to work with the [Python 3 Scikit-Learn Drop-In Environment](../../../public_dropin_environments/python3_sklearn/).

Note that transformers differ from custom models in that the end product is not a trained model, but a transformation of the dataset.
As such, the fit method will train only a preprocessing pipeline on your data. The transform hook must be implemented by the user for now.
This particular transformer acts on both X and y, so you'll need a target (in this case, a regression target). Note that 
since we use a log-transform on y, no training is needed for our y transformer. 

## Instructions:
You should probably change `create_pipeline.py` to contain the code you want. This is where your modeling code will live
You don't need to edit custom.py, but you can if you would like to. It mostly is a shim for communicating with DRUM

## Overview of the pipeline

### Feature identification
- Helper function to identify numeric data types
- Helper function to identify categorical data types

### Preprocessing
Numerics:
- Impute missing values with the median, adding a missing value indicator
- Then center and scale the data

Categoricals:
- Impute missing values with the string "missing"
- One hot encode the data (ignoring new categorical levels at prediction time)

SVD:
After all the above is done, run SVD to reduce the dimensionality of the dataset to 10.

This makes a dataset that can be used with basically any sklearn model.  This step could be removed for models that support sparse data.


### To run locally using 'drum'
Paths are relative to `datarobot-user-models` root:
`drum fit --code-dir model_templates/training/python3_sklearn_transform --input tests/testdata/iris_binary_training.csv --target-type transform --target Species`
If the command succeeds, your code is ready to be uploaded. 
