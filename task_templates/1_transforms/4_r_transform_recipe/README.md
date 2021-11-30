## R Transform Template

This is an R transformer template with preproccessing using a recipe that:
- Handles categorical variables
- Handles numeric variables

This transformer is intended to work with the [R Drop-In Environment](../../../public_dropin_environments/r_lang/).

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
`drum fit --code-dir task_templates/1_transforms/4_r_transform_recipe --input tests/testdata/iris_binary_training.csv --target-type transform --target Species`
If the command succeeds, your code is ready to be uploaded. 

This template can be run with any downstream problem in mind, so the target could be regression, binary, multiclass, or 
there could be no target, making y NULL during fit.
