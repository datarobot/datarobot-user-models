# DataRobot User Models


## Content  
1. [What is this repository?](#what_is_it)
2. [Quickstart and examples](#quickstart)
3. [Assembling an inference model code folder](#inference_model_folder)
4. [Unstructured inference models](#unstructured_inference_models)
5. [Assembling a training model code folder](#training_model_folder)
6. [Custom Model Templates](#custom_model_templates)
7. [Custom Environment Templates](#custom_environment_templates)
8. [Custom Model Runner (drum)](#custom_model_runner)
9. [Contribution & development](#contribution_development)
10. [Communication](#communication)

## What is this repository? <a name="what_is_it"></a>
The **DataRobot User Models** repository contains information and tools for assembling,
debugging, testing, and running your training and inference models with DataRobot.

### Terminology
This repository addresses the DataRobot functionality known as `custom models`. The terms `custom model` and `user model` can be used interchangeably, as can `custom model directory` and `code directory`.

## Quickstart <a name="quickstart"></a>
The following example shows how to use the [DRUM](https://github.com/datarobot/datarobot-user-models/tree/master/custom_model_runner) tool to make predictions on an [sklearn regression model](model_templates/inference/python3_sklearn). For the training model quickstart, please reference [this document](QUICKSTART-FOR-TRAINING.md)
1. Clone the repository
2. Create a virtual environment: `python3 -m virtualenv <dirname for virtual environment>`
3. Activate the virtual environment: `source <dirname for virtual environment>/bin/activate`
4. cd into the repo: `cd datarobot-user-models`
5. Install the required dependencies: `pip install -r public_dropin_environments/python3_sklearn/requirements.txt`
6. Install datarobot-drum: `pip install datarobot-drum`
7. Run the example: `drum score --code-dir model_templates/inference/python3_sklearn --input tests/testdata/boston_housing.csv`  
    > Note: this command assumes model is regression. For binary classification model provide: _**positive-class-label**_ and _**negative-class-label**_ arguments.  
    Input data is expected to be in CSV format. By default, missing values are indicated with NaN in Python, and NA in R according to `pd.read_csv` and `read.csv` respectively.

For more examples, reference the [Custom Model Templates](#custom_model_templates).

## Assembling an inference model code folder <a name="inference_model_folder"></a>
> Note: the following information is only relevant you are using [DRUM](https://github.com/datarobot/datarobot-user-models/tree/master/custom_model_runner) to run a model.

Custom inference models are models trained outside of DataRobot. Once they are uploaded to DataRobot, they are deployed as a DataRobot deployment which supports model monitoring and management.

To create a custom inference model, you must provide specific files to use with a custom environment:

- a serialized model artifact with a file extension corresponding to the chosen environment language.
- any additional custom code required to use it.

The `drum new model` command can help you generate a model template. Check [here](https://github.com/datarobot/datarobot-user-models/tree/master/custom_model_runner#model-template-generation) for more information.

### Built-In Model Support
The DRUM tool has built-in support for the following libraries. If your model is based on one of these libraries, DRUM expects your model artifact to have a matching file extension.

### Python libraries
| Library | File Extension | Example |
| --- | --- | --- |
| scikit-learn | *.pkl | sklean-regressor.pkl |
| xgboost | *.pkl | xgboost-regressor.pkl |
| PyTorch | *.pth | torch-regressor.pth |
| tf.keras (tensorflow>=2.2.1) | *.h5 | keras-regressor.h5 |
| pmml | *.pmml | pmml-regressor.pmml |


### R libraries
| Library | File Extension | Example |
| --- | --- | --- |
| caret | *.rds | brnn-regressor.rds |

This tool makes the following assumptions about your serialized model:
- The data sent to a model can be used to make predictions without additional pre-processing.
- Regression models return a single floating point per row of prediction data.
- Binary classification models return one floating point value <= 1.0 or two floating point values that sum to 1.0 per row of prediction data.
  - Single value output is assumed to be the positive class probability
  - Multi value it is assumed that the first value is the negative class probability, the second is the positive class probability
- There is a single pkl/pth/h5 file present.
- Your model uses one of the above frameworks.

### Custom hooks for Python and R models
If the assumptions mentioned above are incorrect for your model, DRUM supports several hooks for custom code. If needed,
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
- `read_input_data(input_binary_data: bytes) -> Any`
  - `input_binary_data` is a data passed as `--input` parameter in `drum score` mode; or a payload submitted to the `drum server` `/predict` endpoint;
  - If used, this hook must return a non-None value; if it returns something other than a DF, you'll need to write your own score method.
  - This hook can be used to customize data reading, e.g: encoding, handle missing values.
- `load_model(code_dir: str) -> Any`
  - `code_dir` is the directory where the model artifact and additional code are provided, which is passed in the `--code_dir` parameter
  - If used, this hook must return a non-None value
  - This hook can be used to load supported models if your model has multiple artifacts, or for loading models that
  DRUM does not natively support
- `transform(data: DataFrame, model: Any) -> DataFrame`
  - `data` is the dataframe given to DRUM to make predictions on. Missing values are indicated with NaN in Python and NA in R, unless otherwise overridden by the read_input_data hook.
  - `model` is the deserialized model loaded by DRUM or by `load_model` (if provided)
  - This hook is intended to apply transformations to the prediction data before making predictions. It is useful
  if DRUM supports the model's library, but your model requires additional data processing before it can make predictions.
- The transform hook will behave differently when running with a `transform` target type:
    - `transform(X: DataFrame, transformer: Any, y: Series) -> Tuple[DataFrame, Series]`
        - `X` is the dataframe given to DRUM to make predictions on. If `y` is defined, it will contain the raw features to be transformed (omitting the target).
        - `transformer` deserialized transformer loaded by DRUM or `load_model` 
        - `y` defaults to None, and will not be used for unsupervised downstream problems. Contains the target labels. Transform can simply pass the raw values on, may transform the values (i.e perform a log transform), 
        or use the `y` values in the transformation of `X` if needed.
        - As of version 1.4.9, the transform hook need not return `y` if no changes are made to the target. You may define a transform hook that does not take a `y` argument.
- `score(data: DataFrame, model: Any, **kwargs: Dict[str, Any]) -> DataFrame`
  - `data` is the dataframe to make predictions against. If `transform` is supplied, `data` will be the transformed data.
  - `model` is the deserialized model loaded by DRUM or by `load_model`, if supplied
  - `kwargs` - additional keyword arguments to the method; In the case of a binary classification model, contains class labels as the following keys:
    - `positive_class_label` for the positive class label
    - `negative_class_label` for the negative class label
  - This method should return predictions as a dataframe with the following format:
    - Binary classification: requires columns for each class label with floating-point class probabilities as values. Each row
    should sum to 1.0.
    - Regression: requires a single column named `Predictions` with numerical values.
  - This hook is only needed if you would like to use DRUM with a framework not natively supported by the tool.
- `post_process(predictions: DataFrame, model: Any) -> DataFrame`
  - `predictions` is the dataframe of predictions produced by DRUM or by the `score` hook, if supplied.
  - `model` is the deserialized model loaded by DRUM or by `load_model`, if supplied
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
| h2o-genmodel | *.java | GBM_model_python_1589382591366_1.java (pojo)|
| h2o-genmodel | *.zip | GBM_model_python_1589382591366_1.zip (mojo)|
| h2o-genmodel-ext-xgboost | *.java | XGBoost_2_AutoML_20201015_144158.java |
| h2o-genmodel-ext-xgboost | *.zip | XGBoost_2_AutoML_20201015_144158.zip |
| h2o-ext-mojo-pipeline | *.mojo and *.jar | ...|

If you leverage an H2O model exported as POJO, you cannot rename the file.  This does not apply to models exported as MOJO - they may be named in any fashion.

To use h2o-ext-mojo-pipeline, this WILL require an h2o driverless ai license.  

Support for DAI Mojo Pipeline has not be incorporated into tests for the build of `datarobot-drum`

#### Additional params
Define the DRUM_JAVA_XMX environment variable to set JVM maximum heap memory size (-Xmx java parameter), e.g:

```DRUM_JAVA_XMX=512m```

The DRUM tool currently supports models with DataRobot-generated Scoring Code or models that implement either the `IClassificationPredictor`
or `IRegressionPredictor` interface from [datarobot-prediction](https://mvnrepository.com/artifact/com.datarobot/datarobot-prediction).
The model artifact must have a **jar** extension.

## Unstructured inference models <a name="unstructured_inference_models"></a>
**UNDER DEVELOPMENT**
>The feature is under active development.  
>Available in DRUM only (not in DR App).  
>Functionality may change.

Inference models support unstructured mode, where input and output are not verified and can be almost anything.
This is your responsibility to verify correctness.

### Custom hooks for Python and R models
Include any necessary hooks in a file called `custom.py` for Python models or `custom.R` for R models alongside your model artifacts in your model folder:

> Note: The following hook signatures are written with Python 3 type annotations. The Python types match the following R types:
> - None = NULL
> - str = character
> - bytes = raw
> - dict = list
> - tuple = list
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
  DRUM does not natively support
- `score_unstructured(model: Any, data: str/bytes, **kwargs: Dict[str, Any]) -> str/bytes [, Dict[str, str]]`
  - `model` is the deserialized model loaded by DRUM or by `load_model`, if supplied
  - `data` is the data represented as str or bytes, depending on the provided `mimetype`
  - `kwargs` - additional keyword arguments to the method:
    - `mimetype: str` - indicates the nature and format of the data, taken from request Content-Type header or `--content-type` CLI arg in batch mode;
    - `charset: str` - indicates encoding for text data, taken from request Content-Type header or `--content-type` CLI arg in batch mode;
    - `query: dict` - params passed as query params in http request or in CLI `--query` argument in batch mode.
  - This method should return:
    - a single value `return data: str/bytes`
    - a tuple `return data: str/bytes, kwargs: dict[str, str]`
      - where `kwargs = {"mimetype": "users/mimetype", "charset": "users/charset"}` can be used to return mimetype and charset to be used in response Content-Type header.

#### Incoming data type resolution
`score_unstructured` hook receives a `data` parameter which can be of either `str` or `bytes` type.
Type checking methods can be used to verify types:
- in Python`isinstance(data, str)` or `isinstance(data, bytes)`
- in R `is.character(data)` or `is.raw(data)`

DRUM uses `Content-Type` header to determine a type to cast `data` to. Content-Type header can be provided in request or in `--content-type` CLI argument.  
`Content-Type` header format is `type/subtype;parameter`, e.g. `text/plain;charset=utf8`. Only minetype part `text/plain`, and `charset=utf8` parameter matter for DRUM.  
Following rules apply:
- if charset is not defined, default `utf8` charset is used, otherwise provided charset is used to decode data;
- if content type is not defined, then incoming `kwargs={"mimetype": "text/plain", "charset":"utf8"}`, so data is treated as text, decoded using `utf8` charset and passed as `str`;
- if mimetype starts with `text/` or `application/json`, data is treated as text, decoded using provided charset and passed as `str`;
- for all other mimetype values data is treated as binary and passed as `bytes`.

#### Outgoing data and kwargs params
As mentioned above `score_unstructured` can return:
- a single data value `return data`
- a tuple - data and additional params `return data, {"mimetype": "some/type", "charset": "some_charset"}  

##### In server mode
Following rules apply:
- `return data: str` - data is treated as text, default Content-Type=`"text/plain;charset=utf8"` header will be set in response, data encoded using `utf8` charset and sent;
- `return data: bytes` - data is treated as binary, default Content-Type=`"application/octet-stream;charset=utf8"` header will be set in response, data is sent as is;
- `return data, kwargs` - if mimetype value is missing in kwargs, default mimetype will be set according to the data type `str`/`bytes` -> `text/plain`/`application/octet-stream`;
if charset values is missing, default `utf8` charset will be set; then, if data is of type `str`, it will be encoded using resolved charset and sent.

##### In batch mode
The best way to debug in batch mode is to provide `--output` file. Returned data will be written into file according to the type of data returned:
- `str` data -> text file, using default `utf8` or returned in kwargs charset;
- `bytes` data -> binary file.  
(Returned `kwargs` are not shown in the batch mode, but you can still print them during debugging).



## Assembling a training model code folder <a name="training_model_folder"></a>
Custom training models are in active development. They include a `fit()` function, can be trained on the Leaderboard, benchmarked against DataRobot AutoML models, and get access to DataRobot's full set of automated insights. Refer to the [quickrun readme](QUICKSTART-FOR-TRAINING.md).

The model folder must contain any code required for DRUM to run and train your model.

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
The examples in this repository use the DataRobot User Model Runner (DRUM).  For more information on how to use and write models with DRUM, reference the [readme](./custom_model_runner/README.md).

### Sample Models
The [model_templates](model_templates) folder contains sample models that work with the provided template environments. For more information about each model, reference the readme for every example:

##### Inference Models
* [Scikit-Learn sample model](model_templates/inference/python3_sklearn)
* [Scikit-Learn sample unsupervised anomaly detection model](model_templates/inference/python3_sklearn_anomaly)
* [PyTorch sample model](model_templates/inference/python3_pytorch)
* [XGBoost sample model](model_templates/inference/python3_xgboost)
* [Keras sample model](model_templates/inference/python3_keras)
* [Keras sample model + Joblib artifact](model_templates/inference/python3_keras_joblib)
* [PyPMML sample model](model_templates/inference/python3_pmml)
* [R sample model](model_templates/inference/r_lang)
* [Java sample model](model_templates/inference/java_codegen)

##### Training Models
* [Scikit-Learn sample regression model](model_templates/training/python3_sklearn_regression)
* [Scikit-Learn sample binary model](model_templates/training/python3_sklearn_binary)
* [Scikit-Learn sample unsupervised anomaly detection model](model_templates/training/python3_anomaly_detection)
> Note: Unsupervised support is limited to anomaly detection models as of release 1.1.5
* [Scikit-Learn sample transformer](model_templates/training/python3_sklearn_transform)
* [XGBoost sample model](model_templates/training/python3_xgboost)
* [Keras sample model + Joblib artifact](model_templates/training/python3_keras_joblib)
* [R sample model](model_templates/training/r_lang)

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
>Note: DataRobot recommends using an environment template and not building your own environment except for specific use cases. (For example: you don't want to use DRUM but you want to implement your own prediction server.)

If you'd like to use a tool/language/framework that is not supported by our template environments, you can make your own. DataRobot recommends modifying the provided environments to suit your needs. However, to make an easy to use, re-usable environment, you should adhere to the following guidelines:

1) Your environment must include a Dockerfile that installs any requirements you may want.
2) Custom models require a simple webserver in order to make predictions. We recommend putting this in
your environment so that you can reuse it with multiple models. The webserver must be listening on port 8080 and implement the following routes:
   > **Note: `URL_PREFIX` is an environment variable that will be available at runtime. It has to be read and pasted into the routes.**
    1) `GET /URL_PREFIX/` and `GET /URL_PREFIX/ping/` These routes are used to check if your model's server is running.
    2) `GET /URL_PREFIX/health/` This route is used to check if model is loaded and functioning properly.
    3) `POST /URL_PREFIX/predict/` This route is used to make predictions.
3) An executable `start_server.sh` file is required to start the model server.
4) Any code and `start_server.sh` should be copied to `/opt/code/` by your Dockerfile


## Custom Model Runner <a name="custom_model_runner"></a>
Custom model runner (DRUM) is a  tool that helps to assemble, test, and run custom models.
The [custom model runner](custom_model_runner) folder contains its source code.
For more information about how to use it, reference the [pypi docs](https://pypi.org/project/datarobot-drum/).

## Contribution & development <a name="contribution_development"></a>

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

### DR developers
#### DataRobot Confluence
To get more information, search for `custom models` and `datarobot user models` in DataRobot Confluence.

#### Committing into the repo
1. Ask repository admin for write access.
2. Develop your contribution in a separate branch run tests and push to the repository.
3. Create a pull request.

### Non-DataRobot developers
To contribute to the project, use a [regular GitHub process](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork ): fork the repo and create a pull request to the original repository.

### Running tests
*description is being added*

## Communication<a name="communication"></a>
Some places to ask for help are:
- open an issue through the [GitHub board](https://github.com/datarobot/datarobot-user-models/issues).
- ask a question on the [#drum (IRC) channel](https://webchat.freenode.net/?channels=#drum).
