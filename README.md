# DataRobot User Models
## What is it?
**The DataRobot User Models** repository contains boilerplates, information, and tools on how to assemble,
debug, test, and run your training and inference models on the DataRobot platform.

### Terminology
Described DataRobot functionality is known as `Custom Model`, so words `custom model` and `user model` may be interchangeably used.
Also `custom model directory` and `code directory` mean the same entity.


## Content  
- [model templates](model_templates) - contains templates for building and deploying custom models in DataRobot.
- [environment templates](public_dropin_environments) - contains templates of base environments used in DataRobot. Dependency requirements can be applied to the base to create runtime environment for custom models.
- [user models runner](custom_model_runner) - **drum** - a tool that helps to assemble, test, and run custom model. For more information about how to run drum, check out the [pypi docs](https://pypi.org/project/datarobot-drum/)


## Custom Model Templates
This repository contains templates for building and deploying custom models in DataRobot.

Custom Inference Models are models that are trained outside of DataRobot. Once they're uploaded to DR, they are deployed straight to a DR Deployment, and tracked with Model Monitoring and Management.

Custom Training Models are in active development. They include a `fit()` function, and can be trained on the Leaderboard, benchmarked against DR AutoML models, and get access to our full set of automated insights. Check out [The quickrun readme here](QUICKSTART-FOR-TRAINING.md)

### Sample Models
The [model_templates](model_templates) contain example models that will work with the template environments discussed above. For more information about each model,
please see:
##### Inference Models
* [Scikit-Learn sample model](model_templates/inference/python3_sklearn)
* [PyTorch sample model](model_templates/inference/python3_pytorch)
* [XGBoost sample model](model_templates/inference/python3_xgboost)
* [Keras sample model](model_templates/inference/python3_keras)
* [Keras sample model + Joblib artifact](model_templates/inference/python3_keras_joblib)
* [PyPMML sample model](model_templates/inference/python3_pmml)
* [R sample model](model_templates/r_lang)
* [Java sample model](model_templates/java_codegen)

##### Training Models
* [Scikit-Learn sample model](model_templates/training/python3_sklearn)
* [XGBoost sample model](model_templates/training/python3_xgboost)
* [Keras sample model + Joblib artifact](model_templates/training/python3_keras_joblib)


## Custom Environment Templates
A custom environment defines the runtime environment for a custom model.  In this repository, we provide several example environments that you can use and modify:
* [Python 3 + sklearn](public_dropin_environments/python3_sklearn)
* [Python 3 + PyTorch](public_dropin_environments/python3_pytorch)
* [Python 3 + xgboost](public_dropin_environments/python3_xgboost)
* [Python 3 + keras/tensorflow](public_dropin_environments/python3_keras)
* [Python 3 + pmml](public_dropin_environments/python3_pmml)
* [R + caret](public_dropin_environments/r_lang)
* [Java Scoring Code](public_dropin_environments/java_codegen)

These sample environments each define the libraries available in the environment and are designed to allow for simple custom models to be made that consist solely of your model's artifacts and an optional custom code
file, if necessary.

For detailed information on how to create models that work in these environments, please click on the links above for the relevant environment.

## Building your own environment
If you'd like to use a tool/language/framework that is not supported by our template environments, you can make your own. We recommend modifying the provided environments to suit your needs,
but to make an easy to use, reusable environment in general, you should follow the following guidelines/requirements:

1) Your environment must include a Dockerfile that installs any requirements you may want.
1) Custom models require a simple webserver in order to make predictions. We recommend putting this in
your environment so that you can reuse it with multiple models. The webserver must  be listening on port 8080 and implement the following routes:
    1) `GET /{URL_PREFIX}/` This route is used to check if your model's server is running
    1) `POST /URL_PREFIX/predict/` This route is used to make predictions
1) An executablle `start_server.sh` file is required and should start the model server
1) Any code and start_server.sh should be copied to `/opt/code/` by your Dockerfile
> **_NOTE:_** URL_PREFIX is an environment variable that will be available at runtime

## Contribution

### Prerequisites for development
This section is only relevant if you plan to work on the **drum**
To build it, following packages are required:
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

### DR developers
#### DataRobot Confluence
To get more information, search for `custom models` and `datarobot user models` in DataRobot Confluence.

#### Committing into the repo
1. Ask repository admin for write access.
2. Develop your contribution in a separate branch run tests and push to the repository.
3. Create a pull request

### Non-DataRobot developers
To contribute to the project use a regular GitHub process: fork the repo -> create a pull request to the original repository.
https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork 

### Report bugs
To report a bug, open an issue through GitHub board
https://github.com/datarobot/datarobot-user-models/issues

### Running tests
*description is being added*
