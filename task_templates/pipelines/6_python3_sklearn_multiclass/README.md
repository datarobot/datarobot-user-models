## Advanced Python Template Model

This is an advanced python task template with preproccessing that:
- Handles text variables
- Handles categorical variables
- Handles numeric variables

The final estimator can be any sklearn-compatible estimator.

This model is intended to work with the [Python 3 Scikit-Learn Drop-In Environment](../../../public_dropin_environments/python3_sklearn/).

## Instructions:
You should probably change `create_pipeline.py` to contain the code you want. This is where your modeling code will live
You don't need to edit custom.py, but you can if you would like to. It mostly is a shim for communicating with DRUM

## Overview of the pipeline

### Feature identification
- Helper function to identify numeric data types
- Helper function to identify categorical data types
- Helper function to identify text data types (This is pretty cool, and demonstrates how one might write more advanced column selectors for complicated blueporints)

### Preprocessing
Numerics:
- Impute missing values with the median, adding a missing value indicator
- Then center and scale the data

Categoricals:
- Impute missing values with the string "missing"
- One hot encode the data (ignoring new categorical levels at prediction time)

Text:
- Impute missing values with the string "missing"
- Tfidf encode the text, using 1-grams and 2-grams.

SVD:
After all the above is done, run SVD to reduce the dimensionality of the dataset to 10.

This makes a dataset that can be used with basically any sklearn model.  This step could be removed for models that support sparse data.

### Modeling
Penalized linear model (but you can drop in any other sklearn model)


### To run locally using 'drum'
Paths are relative to `datarobot-user-models` root:
`drum fit --code-dir task_templates/pipelines/python3_sklearn_multiclass --input tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv --target-type multiclass --target class`
If the command succeeds, your code is ready to be uploaded.
