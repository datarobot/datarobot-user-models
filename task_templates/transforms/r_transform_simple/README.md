## R Transform Template

This is an R transformer template with preproccessing that:
- Handles numeric variables

This transformer is intended to work with the [R Drop-In Environment](../../../public_dropin_environments/r_lang/).

## Instructions:
You can change the contents of the `fit` hook to contain the code you want. The `transform` hook will ingest the artifact
created from fit and use it to transform the input data.


### Preprocessing
Numerics:
- Impute missing values with the median

This makes a dataset that can be used with any model that supports dense numerics.

### To run locally using 'drum'
Paths are relative to `datarobot-user-models` root:
`drum fit --code-dir task_templates/transforms/r_transform_simple --input tests/testdata/iris_binary_training.csv --target-type transform --target Species`
If the command succeeds, your code is ready to be uploaded. 

This template can be run with any downstream problem in mind, so the target could be regression, binary, multiclass, or 
there could be no target, making y NULL during fit.
