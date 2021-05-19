# Julia Drop-In Template Environment

This template environment can be used to create artifact-only Julia MLJ models or models with which MLJ can interface.

Your custom model directory needs only contain your model artifact if you use the environment correctly.

## Supported Libraries

This environment has built for Julia and has support for the following scientific libraries.  If you leverage the MLJ interface for the following libraries, then you can take the serialized model with `jlso` extension and handle those with no additional lift.  

- MLJ
- ScikitLearn
- DecisionTree
- Clustering
- GLM
- LIBSVM
- MultivariateStats
- ScikitLearn
- XGBoost

## Instructions

1. From the terminal, run `tar -czvf py_dropin.tar.gz -C /path/to/public_dropin_environments/julia_mlj/ .`
2. Using either the API or from the UI create a new Custom Environment with the tarball created
in step 1.

### Creating models for this environment

To use this environment, your custom model archive must contain a single serialized model artifact
with `.jlso` file extension as well as any other custom code needed to use your serialized model.


This environment makes the following assumption about your serialized model:
- The data sent to custom model can be used to make predictions without
additional pre-processing
- Regression models return a single floating point per row of prediction data
- Binary classification models return two floating point values that sum to 1.0 per row of prediction data.
  - The first value is the negative class probability, the second is the positive class probability
- There is a single jlso file present
  
If these assumptions are incorrect for your model, you should make a copy of [custom.jl](need link), modify it as needed, and include in your custom model archive.

The structure of your custom model archive should look like:

- custom_model.tar.gz
  - artifact.jlso
  - custom.jl (if needed)

Please read [datarobot-cmrunner](../../custom_model_runner/README.md) documentation on how to assemble **custom.jl**.