dummy
# DataRobot User Models

## What is this repository? <a name="what_is_it"></a>
This repository contains tools, templates, and information for assembling, debugging, testing, 
and running your custom inference models and custom tasks with DataRobot.

The [./task_templates](./task_templates) 
and [./model_templates](./model_templates)
folders provide reference examples to
help users learn how to create custom tasks and/or custom inference models. 
The templates there are simple, well documented, and can be used as tutorials. 
These templates should also remain up to date with any API or other changes.

For further examples, provided as-is, that often contain more complex logic please
see the community examples repo at: https://github.com/datarobot-community/custom-models. 
Please note that these examples may not stay up to date with the latest API
or best practices. 


For further documentation on this and all other features please visit our 
comprehensive documentation at: https://docs.datarobot.com/

## Terminology
DataRobot has 2 mechanisms for bringing custom ML code:

1. Custom task: an ML algorithm, for example, XGBoost or One-hot encoding, 
   that can be used as a step in an ML pipeline ([blueprint](https://docs.datarobot.com/en/docs/modeling/analyze-models/describe/blueprints.html)) 
   inside DataRobot.
   
2. Custom inference model: a pre-trained model or user code prepared for inference. 
   An inference model can have a predefined input/output schema or be unstructured. 
   Learn more [here](https://docs.datarobot.com/en/docs/mlops/deployment/custom-models/index.html)

## Content

1. [Custom Tasks Reference](#custom_task_ref)
2. [Custom Inference Model Reference](#custom_inference_model_ref)
3. [Contribution and development](#contribution)
4. [Communication](#communication)

## Custom Tasks Reference <a name="custom_task_ref"></a>

Materials for getting started:

* [Demo Video](https://youtu.be/XvtARLw8zVo)
* Code examples:
  * [Custom task templates](https://github.com/datarobot/datarobot-user-models/tree/master/task_templates)
  * [Environment Templates](https://github.com/datarobot/datarobot-user-models/tree/master/public_dropin_environments)
  * [Building blueprints programmatically](https://blueprint-workshop.datarobot.com/)
    from tasks like lego blocks     
* [Quick walk-through](https://docs.datarobot.com/en/docs/release/public-preview/automl-preview/cml/cml-quickstart.html)
* [Detailed documentation](https://docs.datarobot.com/en/docs/release/public-preview/automl-preview/cml/index.html)

Other resources:
* There is a chance that the task you are looking for has already been implemented. 
  Check [custom tasks community Github](https://github.com/datarobot-community/custom-models/tree/master/custom_tasks) 
  to see some off-the-shelf examples
  * Note: The community repo above is NOT the place to start learning the basic concepts. 
    The examples tend to have more complex logic and are meant to be used 
    as-is rather than as a reference.
  * [This repo](task_templates)
    is the appropriate place to start with tutorial examples.

## Custom Inference Models Reference <a name="custom_inference_model_ref"></a>

Materials for getting started:

* [Custom models walk-through](https://community.datarobot.com/t5/knowledge-base/working-with-custom-models/ta-p/6082)
* Code examples:
    * [Custom inference models templates](https://github.com/datarobot/datarobot-user-models/tree/master/model_templates)
    * [Environment Templates](https://github.com/datarobot/datarobot-user-models/tree/master/public_dropin_environments)
* References for defining a custom inference model:
    * [Assemble a code folder](DEFINE-INFERENCE-MODEL.md#inference_model_folder)
    * [Define a structured model](DEFINE-INFERENCE-MODEL.md#structured_inference_model)
    * [Define an unstructured model](DEFINE-INFERENCE-MODEL.md#unstructured_inference_model)
    * [Test a model with DRUM locally](DEFINE-INFERENCE-MODEL.md#test_inference_model_drum)
    * [Build your own environment](DEFINE-INFERENCE-MODEL.md#build_own_environment)
    * [Upload, manage, and deploy a model in DataRobot](DEFINE-INFERENCE-MODEL.md#upload_custom_model)

Other sources:
* There is a chance that the model you are looking for has already been implemented. 
  Check [custom inference models community Github](https://github.com/datarobot-community/custom-models/tree/master/custom_inference) 
  to see some off-the-shelf examples
    


## Contribution & development <a name="contribution"></a>

### Prerequisites for development
> Note: Only reference this section if you plan to work with DRUM.

To build it, the following packages are required:
`make`, `Java 11`, `maven`, `docker`, `R`
E.g. for Ubuntu 18.04  
`apt-get install build-essential openjdk-11-jdk openjdk-11-jre maven python3-dev docker apt-utils curl gpg-agent software-properties-common dirmngr libssl-dev ca-certificates locales libcurl4-openssl-dev libxml2-dev libgomp1 gcc libc6-dev pandoc`

#### R
Ubuntu 18.04  
`apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9`  
`add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'`  
`apt-get install r-cran-littler r-base r-base-dev`  
#### R packages
`Rscript -e "install.packages(c('devtools', 'tidyverse', 'caret', 'recipes', 'glmnet', 'plumber', 'Rook', 'rjson', 'e1071'), Ncpus=4)"`  
`Rscript -e 'library(caret); install.packages(unique(modelLookup()[modelLookup()$forReg, c(1)]), Ncpus=4)'`  
`Rscript -e 'library(caret); install.packages(unique(modelLookup()[modelLookup()$forClass, c(1)]), Ncpus=4)'`

### DRUM developers

#### Setting Up Local Env For Testing

1. create Py 3.7 or 3.8 venv
1. `pip install -r requirements_dev.txt`
1. `pip install -e custom_model_runner/`
1. pytest to your heart's content

#### DataRobot Confluence
To get more information, search for `custom models` and `datarobot user models` in DataRobot Confluence.

#### Committing into the repo
1. Ask repository admin for write access.
2. Develop your contribution in a separate branch run tests and push to the repository.
3. Create a pull request.

#### Testing changes to drum in DR app
There is a script called `create-drum-dev-image.sh` which will build and save an image with your latest local changes to the DRUM codebase. You can test new changes to drum in the DR app by running this script with an argument for which dropin env to modify, and uploading the image which gets built as an execution environment. 

### Non-DataRobot developers
To contribute to the project, use a [regular GitHub process](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork ): fork the repo and create a pull request to the original repository.

### Tests

#### Test artifacts
Artifacts used in tests are located here: [./tests/fixtures/drop_in_model_artifacts](./tests/fixtures/drop_in_model_artifacts).  
There is also the code in (*.ipynb, Pytorch.py, Rmodel.R, etc files) to generate those artifacts.  
Check for `generate*` scripts in [./tests/fixtures/drop_in_model_artifacts](./tests/fixtures/drop_in_model_artifacts) and [./tests/fixtures/artifacts.py](./tests/fixtures/artifacts.py)

Model examples in [./model_templates](./model_templates) are also used in functional testing. In the most cases, artifacts for those models are the same as in the [./tests/fixtures/drop_in_model_artifacts](./tests/fixtures/drop_in_model_artifacts) and can be simply copied accordingly.
If artifact for model template is not in the [./tests/fixtures/drop_in_model_artifacts](./tests/fixtures/drop_in_model_artifacts), check template's README for more instructions.


## Communication<a name="communication"></a>
Some places to ask for help are:
- open an issue through the [GitHub board](https://github.com/datarobot/datarobot-user-models/issues).
