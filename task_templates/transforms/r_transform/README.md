## R Transform Template

This is an R transformer template with preproccessing that:
- Handles categorical variables
- Handles numeric variables

This transformer is intended to work with the [R Drop-In Environment](../../../public_dropin_environments/r_lang/).

Note that transformers differ from custom models in that the end product is not a trained model, but a transformation of the dataset.
As such, the fit method will train only a preprocessing pipeline on your data.

## Instructions:
You can change the contents of the `fit` hook to contain the code you want. The `transform` hook will simply 'bake the recipe'
(i.e. transform using the pipeline) automatically. If you choose not to use a recipe as an artifact, then you must fill out the
transform hook.


### Preprocessing
- Drop constant columns

Numerics:
- Impute missing values with the median, adding a missing value indicator
- Then center and scale the data

Categoricals:
- Pool infrequent occurring values into an "other" category
- One hot encode the data (ignoring new categorical levels at prediction time)

This makes a dataset that can be used with any model that supports sparse data.

### To run locally using 'drum'
Paths are relative to `datarobot-user-models` root:
`drum fit --code-dir task_templates/transforms/r_transform --input tests/testdata/iris_binary_training.csv --target-type transform --target Species`
If the command succeeds, your code is ready to be uploaded. 

This template can be run with any downstream problem in mind, so the target could be regression, binary, multiclass, or 
there could be no target and the preprocessing will treat the full training data as an X matrix.
