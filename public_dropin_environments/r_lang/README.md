# R Drop-In Template Environment

This template environment can be used to create artifact-only R custom models that use the caret library.
Your custom model directory needs only contain your model artifacts if you use the
environment correctly.

## Supported Libraries

This environment has built for R with support for the [caret](http://topepo.github.io/caret/index.html)
library of models.

### Other dependencies
This environment uses **cmrun** to run custom model.
**cmrun** uses `rpy2` package (by default the latest version is installed) to run R.
You may need to adjust **rpy2** and **pandas** versions for compatibility.

### Supported Model Types

Due to the time required to install all libraries recommended by caret, only model types that are also package names are installed (ex. `brnn`, `glmnet`). If you would like to use models that require additional packages, you will need to make a copy of this environment and modify the Dockerfile to install additional packages.  To speed up the build time, you can remove the lines in the `# Install caret models` section and install only what you need.

To check if your model's method matches its package name, please refer to the official [docs](http://topepo.github.io/caret/available-models.html)

## Instructions

1. From the terminal, run `tar -czvf r_dropin.tar.gz -C /path/to/public_dropin_enironments/r_lang/ .`
2. Using either the API or from the UI create a new Custom Environment with the tarball created
in step 1.

> **_NOTE_** This environment may take more than an hour to build.

### Creating models for this environment

To use this environment, your custom model archive must contain a serialized model artifact
as an RDS file with the file extension `.rds` (ex. `trained_glm.rds`), as well as any other custom code
and artifacts needed to use your serialized model.

This environment makes the following assumption about your serialized model:
- The data sent to custom model can be used to make predictions without
additional pre-processing
- No additional libraries need to be loaded in order to properly use your model.
- Regression models return a single floating point per row of prediction data
- Binary classification models return two floating point values that sum to 1.0 per row of prediction data.
  - The first value is the negative class probability, the second is the positive class probability
- There is a single rds file present
  
If these assumptions are incorrect for your model, you should make a copy of [custom.R](https://github.com/datarobot/datarobot-user-models/blob/master/model_templates/inference/r_lang/custom.R), modify it as needed, and include it in your custom model archive.

The structure of your custom model archive should look like:

- custom_model.tar.gz
  - artifact.rds
  - custom.R (if needed)

Please read [datarobot-cmrunner](../../custom_model_runner/README.md) documentation on how to assemble **custom.R**.
