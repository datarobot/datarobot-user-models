## Sklearn Transform with Hyperparameters Template

This is a numeric python transformer template that showcases how to use hyperparameters. It performs
preproccessing that:
- Imputes missing values
- Bins numerics, represented as a onehot encoded sparse matrix

This transformer is intended to work with the [Python 3 Scikit-Learn Drop-In Environment](../../../public_dropin_environments/python3_sklearn/).

Defining and using hyperparameters are crucial for speeding up experimentation - they allow you to upload your
custom task just once, and reuse that version to modify model parameters.

## Instructions:
`model-metadata.yaml` showcases how hyperparameters are defined. This includes the parameter names, types, and 
ranges allowed for each parameter. Then within DataRobot's Composable ML editor or advance tuning, you can set the 
parameter values. `custom.py`'s fit function showcases how parameters are passed and used within a custom task. 

### To run locally using 'drum'
TODO: [MODEL-7902] update path once its moved into the template folder 

Paths are relative to `datarobot-user-models` root:
`drum fit --code-dir task_templates/1_transforms/python3_sklearn_transform --input tests/testdata/iris_binary_training.csv --target-type transform --target Species`
If the command succeeds, your code is ready to be uploaded. `drum fit` will pass in the default hyperparameter values.

This template can be run with any downstream problem in mind, so the target could be regression, binary, multiclass, or 
there could be no target and the preprocessing will treat the full training data as an X matrix.
