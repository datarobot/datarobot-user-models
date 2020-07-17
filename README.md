# DataRobot User Models


## Content  
1. [What is this repository?](#what_is_it)
2. [Quickstart and examples](#quickstart)
3. [Assembling an inference model code folder](#inference_model_folder)
4. [Assembling a training model code folder](#training_model_folder)
5. [Custom Model Templates](#custom_model_templates)
6. [Custom Environment Templates](#custom_environment_templates)
7. [Custom Model Runner (drum)](#custom_model_runner)
8. [Contribution & development](#contribution_development)

## What is this repository? <a name="what_is_it"></a>
The **DataRobot User Models** repository contains information and tools for assembling,
debugging, testing, and running your training and inference models with DataRobot.

#### Terminology
This repository address the DataRobot functionality known as `custom models`. The terms `custom model` and `user model` can be used interchangeably, as can `custom model directory` and `code directory`.

## Quickstart <a name="quickstart"></a>
The following example shows how to use the [**drum**](https://github.com/datarobot/datarobot-user-models/tree/master/custom_model_runner) tool to make predictions on an [sklearn regression model](model_templates/inference/python3_sklearn)
1. Clone the repository
2. Create a virtual environment: `python3 -m virtualenv <dirname for virtual environment>`
3. Activate the virtual environment: `source <dirname for virtual environment>/bin/activate`
4. cd into the repo: `cd datarobot-user-models`
5. Install the required dependencies: `pip install -r public_dropin_environments/python3_sklearn/requirements.txt`
6. Install datarobot-drum: `pip install datarobot-drum`
7. Run the example: `drum score --code-dir model_templates/inference/python3_sklearn --input tests/testdata/boston_housing.csv`  
    > Note: this command assumes model is regression. For binary classification model provide: _**positive-class-label**_ and _**negative-class-label**_ arguments.

For more examples, reference the [Custom Model Templates](#custom_model_templates).

## Assembling an inference model code folder <a name="inference_model_folder"></a>
> Note: the following information is only relevant you are using [**drum**](https://github.com/datarobot/datarobot-user-models/tree/master/custom_model_runner) to run a model. 

Custom inference models are models trained outside of DataRobot. Once they are uploaded to DataRobot, they are deployed as a DataRobot deployment which supports model monitoring and management.

To create a custom inference model, you must provide specific files to use with a custom environment:

- a serialized model artifact with a file extension corresponding to the chosen environment language.
- any additional custom code required to use it.

The `drum new model` command can help you generate a model template. Check [here](https://github.com/datarobot/datarobot-user-models/tree/master/custom_model_runner#model-template-generation) for more information.

### Built-In Model Support
The **drum** tool has built-in support for the following libraries. If your model is based on one of these libraries, **drum** expects your model artifact to have a matching file extension.

### Python libraries
| Library | File Extension | Example |
| --- | --- | --- |
| scikit-learn | *.pkl | sklean-regressor.pkl |
| xgboost | *.pkl | xgboost-regressor.pkl |
| PyTorch | *.pth | torch-regressor.pth |
| keras | *.h5 | keras-regressor.h5 |
| pmml | *.pmml | pmml-regressor.pmml |


### R libraries
| Library | File Extension | Example |
| --- | --- | --- |
| caret | *.rds | brnn-regressor.rds |

This tool makes the following assumptions about your serialized model:
- The data sent to a model can be used to make predictions without additional pre-processing.
- Regression models return a single floating point per row of prediction data.
- Binary classification models return two floating point values that sum to 1.0 per row of prediction data.
  - The first value is the positive class probability, the second is the negative class probability
- There is a single pkl/pth/h5 file present.
- Your model uses one of the above frameworks.
  
### Custom hooks for Python and R models
If the assumptions mentioned above are incorrect for your model, **drum** supports several hooks for custom code. If needed,
include any necessary hooks in a file called `custom.py` for Python models or `custom.R` for R models alongside your model artifacts in your model folder:

> Note: The following hook signatures are written with Python 3 type annotations. The Python types match the following R types:
> - DataFrame = data.frame
> - None = NULL
> - str = character
> - Any = R Object (the deserialized model)
> - *args, **kwargs = ... (these aren't types, they're just placeholders for additional parameters)

- `init(**kwargs) -> None`
  - Executed once in the beginning of the run
  - `kwargs` - additional keyword arguments to the method;
    - code_dir - code folder passed in the `--code_dir` parameter
- `load_model(code_dir: str) -> Any`
  - `code_dir` is the directory where the model artifact and additional code are provided, which is passed in the `--code_dir` parameter
  - If used, this hook must return a non-None value
  - This hook can be used to load supported models if your model has multiple artifacts, or for loading models that
  **drum** does not natively support
- `transform(data: DataFrame, model: Any) -> DataFrame`
  - `data` is the dataframe given to **drum** to make predictions on
  - `model` is the deserialized model loaded by **drum** or by `load_model` (if provided)
  - This hook is intended to apply transformations to the prediction data before making predictions. It is useful
  if **drum** supports the model's library, but your model requires additional data processing before it can make predictions.
- `score(data: DataFrame, model: Any, **kwargs: Dict[str, Any]) -> DataFrame`
  - `data` is the dataframe to make predictions against. If `transform` is supplied, `data` will be the transformed data.
  - `model` is the deserialized model loaded by **drum** or by `load_model`, if supplied
  - `kwargs` - additional keyword arguments to the method; In the case of a binary classification model, contains class labels as the following keys:
    - `positive_class_label` for the positive class label
    - `negative_class_label` for the negative class label
  - This method should return predictions as a dataframe with the following format:
    - Binary classification: requires columns for each class label with floating-point class probabilities as values. Each row
    should sum to 1.0.
    - Regression: requires a single column named `Predictions` with numerical values.
  - This hook is only needed if you would like to use **drum** with a framework not natively supported by the tool.
- `post_process(predictions: DataFrame, model: Any) -> DataFrame`
  - `predictions` is the dataframe of predictions produced by **drum** or by the `score` hook, if supplied.
  - `model` is the deserialized model loaded by **drum** or by `load_model`, if supplied
  - This method should return predictions as a dataframe with the following format:
    - Binary classification: requires columns for each class label with floating-point class probabilities as values. Each row
    should sum to 1.0.
    - Regression: requires a single column called `Predictions` with numerical values.
  - This method is only needed if your model's output does not match the above expectations.
> Note: training and inference hooks can be defined in the same file.

### Java
| Library | File Extension | Example |
| --- | --- | --- |
| datarobot-prediction | *.jar | dr-regressor.jar |

#### Additional params
Define the DRUM_JAVA_XMX environment variable to set JVM maximum heap memory size (-Xmx java parameter), e.g:

```DRUM_JAVA_XMX=512m```

The **drum** tool currently supports models with DataRobot-generated Scoring Code or models that implement either the `IClassificationPredictor`
or `IRegressionPredictor` interface from [datarobot-prediction](https://mvnrepository.com/artifact/com.datarobot/datarobot-prediction).
The model artifact must have a **jar** extension.

## Assembling a training model code folder <a name="training_model_folder"></a>
Custom training models are in active development. They include a `fit()` function, can be trained on the Leaderboard, benchmarked against DataRobot AutoML models, and get access to DataRobot's full set of automated insights. Refer to the [quickrun readme](QUICKSTART-FOR-TRAINING.md).

The model folder must contain any code required for **drum** to run and train your model.

### Python
The model folder must contain a `custom.py` file which defines a `fit` method.

- `fit(X: pandas.DataFrame, y: pandas.Series, output_dir: str, **kwargs: Dict[str, Any]) -> None`
    - `X` is the dataframe to perform fit on.
    - `y` is the dataframe containing target data.
    - `output_dir` is the path to write the model artifact to.
    - `kwargs` additional keyword arguments to the method;
        - `class_order: List[str]` a two element long list dictating the order of classes which should be used for modeling.
        - `row_weights: np.ndarray` an array of non-negative numeric values which can be used to dictate how important a row is.

> Note: Training and inference hooks can be defined in the same file.


## Custom Model Templates <a name="custom_model_templates"></a>
The [model templates](model_templates) folder provides templates for building and deploying custom models in DataRobot. Use the templates as an example structure for your own custom models.

### DataRobot User Model Runner
The examples in this repository use the DataRobot User Model Runner (**drum**).  For more information on how to use and write models with **drum**, reference the [readme](./custom_model_runner/README.md).

### Sample Models
The [model_templates](model_templates) folder contains sample models that work with the provided template environments. For more information about each model, reference the readme for every example:

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


## Custom Environment Templates <a name="custom_environment_templates"></a>
The [environment templates folder](#custom_environment_template) contains templates for the base environments used in DataRobot. Dependency requirements can be applied to the base environment to create a runtime environment for custom models.
A custom environment defines the runtime environment for a custom model. In this repository, we provide several example environments that you can use and modify:
* [Python 3 + sklearn](public_dropin_environments/python3_sklearn)
* [Python 3 + PyTorch](public_dropin_environments/python3_pytorch)
* [Python 3 + xgboost](public_dropin_environments/python3_xgboost)
* [Python 3 + keras/tensorflow](public_dropin_environments/python3_keras)
* [Python 3 + pmml](public_dropin_environments/python3_pmml)
* [R + caret](public_dropin_environments/r_lang)
* [Java Scoring Code](public_dropin_environments/java_codegen)

These sample environments each define the libraries available in the environment and are designed to allow for simple custom models to be made that consist solely of your model's artifacts and an optional custom code
file, if necessary.

For detailed information on how to create models that work in these environments, reference the links above for each environment.

## Building your own environment
>Note: DataRobot recommends using an environment template and not building your own environment except for specific use cases. (For example: you don't want to use **drum** but you want to implement your own prediction server.)

If you'd like to use a tool/language/framework that is not supported by our template environments, you can make your own. DataRobot recommends modifying the provided environments to suit your needs. However, to make an easy to use, re-usable environment, you should adhere to the following guidelines:

1) Your environment must include a Dockerfile that installs any requirements you may want.
2) Custom models require a simple webserver in order to make predictions. We recommend putting this in
your environment so that you can reuse it with multiple models. The webserver must be listening on port 8080 and implement the following routes:
    1) `GET /{URL_PREFIX}/` This route is used to check if your model's server is running.
    2) `POST /URL_PREFIX/predict/` This route is used to make predictions.
3) An executable `start_server.sh` file is required to start the model server.
4) Any code and `start_server.sh` should be copied to `/opt/code/` by your Dockerfile
> Note: `URL_PREFIX` is an environment variable that will be available at runtime.

## Custom Model Runner <a name="custom_model_runner"></a>
Custom model runner (**drum**) is a  tool that helps to assemble, test, and run custom models.
The [custom model runner](custom_model_runner) folder contains its source code. 
For more information about how to use it, reference the [pypi docs](https://pypi.org/project/datarobot-drum/).

## Contribution & development <a name="contribution_development"></a> 

### Prerequisites for development
> Note: Only reference this section if you plan to work with **drum**.

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

### DR developers
#### DataRobot Confluence
To get more information, search for `custom models` and `datarobot user models` in DataRobot Confluence.

#### Committing into the repo
1. Ask repository admin for write access.
2. Develop your contribution in a separate branch run tests and push to the repository.
3. Create a pull request.

### Non-DataRobot developers
To contribute to the project, use a [regular GitHub process](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork ): fork the repo and create a pull request to the original repository.


### Report bugs
To report a bug, open an issue through the [GitHub board](https://github.com/datarobot/datarobot-user-models/issues).


### Running tests
*description is being added*
