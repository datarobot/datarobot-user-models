## Sklearn Transform Template

This is a python transformer template with preproccessing that:
- Handles categorical variables
- Handles numeric variables

This transformer is intended to work with the [Python 3 Scikit-Learn Drop-In Environment](../../../public_dropin_environments/python3_sklearn/).

Note that transformers differ from custom models in that the end product is not a trained model, but a transformation of the dataset.
As such, the fit method will train only a preprocessing pipeline on your data. The transform hook must be implemented by the user for now.
You will also have to pass target info according to the downstream modeling task that will be used by this transformer.

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

This makes a dataset that can be used with any sklearn model that supports sparse data.

### To run locally using 'drum'
Paths are relative to `datarobot-user-models` root:
`drum fit --code-dir task_templates/1_transforms/3_python3_sklearn_transform --input tests/testdata/iris_binary_training.csv --target-type transform --target Species`
If the command succeeds, your code is ready to be uploaded. 

This template can be run with any downstream problem in mind, so the target could be regression, binary, multiclass, or 
there could be no target and the preprocessing will treat the full training data as an X matrix.
