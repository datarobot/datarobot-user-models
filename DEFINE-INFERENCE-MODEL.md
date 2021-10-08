# How to define a custom inference model 

Note: See overview at [README](https://github.com/datarobot/datarobot-user-models#readme)

## Content

1. [Assembling an inference model code folder](#inference_model_folder)
2. [Define structured inference model](#structured_inference_model)
3. [Define unstructured inference model](#unstructured_inference_model)
4. [Test an inference model with DRUM locally](#test_inference_model_drum)
5. [Building your own environment](#build_own_environment)
6. [Upload, manage, and deploy a custom inference model with DataRobot](#upload_custom_model)
    1. see [here](https://app.datarobot.com/docs/mlops/deployment/custom-models/custom-inf-model.html)
    as well for documentation on how to upload, manage, and deploy custom inference models

## Assembling an inference model code folder <a name="inference_model_folder"></a>

> Note: the following information is only relevant you are using
[DRUM](https://github.com/datarobot/datarobot-user-models/tree/master/custom_model_runner)
to run a model.

Custom inference models are models trained outside DataRobot.
Once they are uploaded to DataRobot, they are deployed as a DataRobot deployment which
supports model monitoring and management.

To create a custom inference model, you must provide specific files to use with a
custom environment:

- a serialized model artifact with a file extension corresponding to the chosen environment language.
- any additional custom code required to use it.

The `drum new model` command can help you generate a model template.
Check [here](https://github.com/datarobot/datarobot-user-models/tree/master/custom_model_runner#model-template-generation) for more information.

See [here](https://github.com/datarobot/datarobot-user-models/tree/master/model_templates) 
for examples you can use as a template. 

## Define a structured inference model <a name="structured_inference_model"></a>

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

### Julia Libraries
| Library | File Extension | Example |
| --- | --- | --- |
| MLJ | *.jlso | grade_regression.jlso |

This tool makes the following assumptions about your serialized model:
- The data sent to a model can be used to make predictions without additional pre-processing.
- Regression models return a single floating point per row of prediction data.
- Binary classification models return one floating point value <= 1.0 or two floating point values that sum to 1.0 per row of prediction data.
    - Single value output is assumed to be the positive class probability
    - Multi value it is assumed that the first value is the negative class probability, the second is the positive class probability
- There is a single pkl/pth/h5 file present.
- Your model uses one of the above frameworks.

### Data format
When working with structured models DRUM supports data as files of `csv`, `sparse`, or `arrow` format.   
DRUM doesn't perform sanitation of missing or strange(containing parenthesis, slash, etc symbols) column names.

### Custom hooks for Python and R models
If the assumptions mentioned above are incorrect for your model, DRUM supports several hooks for custom code. If needed,
include any necessary hooks in a file called `custom.py` for Python models,`custom.R` for R models, or `custom.jl  for Julia models alongside your model artifacts in your model folder:

> Note: The following hook signatures are written with Python 3 type annotations. The Python types match the following R types and Julia types:
> - DataFrame = data.frame = DataFrames.DataFrame
> - None = NULL = nothing
> - str = character = String
> - Any = R Object (the deserialized model)
> - *args, **kwargs = ... = kwargs** (these aren't types, they're just placeholders for additional parameters)

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
    - This hook can be used in both transformer and estimator tasks.
        - For transformers, it will apply the transformations to X and pass it to downstream tasks.
        - For estimators, it is intended to apply transformations to the prediction data before making predictions.
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

#### Details on Julia

Julia models are supported by usage of [pyjulia](https://pyjulia.readthedocs.io/en/latest/api.html).

pyjulia does NOT work with statically linked libpython.  See this trouble shooting [article](https://pyjulia.readthedocs.io/en/latest/troubleshooting.html).  Other issues arise when using a different versions of python to build Julia PyCall vs what currently being called with (often seen with pyenv).

The simplest way to getting DRUM working with Julia is to leverage the [Julia Dropin Environment](public_dropin_environments/julia_mlj)

See details [here](model_templates/julia/README.md) on setting up Julia for use with DRUM

### Java
| Library | File Extension | Example |
| --- | --- | --- |
| datarobot-prediction | *.jar | dr-regressor.jar |
| h2o-genmodel | *.java | GBM_model_python_1589382591366_1.java (pojo)|
| h2o-genmodel | *.zip | GBM_model_python_1589382591366_1.zip (mojo)|
| h2o-genmodel-ext-xgboost | *.java | XGBoost_2_AutoML_20201015_144158.java |
| h2o-genmodel-ext-xgboost | *.zip | XGBoost_2_AutoML_20201015_144158.zip |
| h2o-ext-mojo-pipeline | *.mojo | ...|

If you leverage an H2O model exported as POJO, you cannot rename the file.  This does not apply to models exported as MOJO - they may be named in any fashion.

To use h2o-ext-mojo-pipeline, this WILL require an h2o driverless ai license.

Support for DAI Mojo Pipeline has not be incorporated into tests for the build of `datarobot-drum`

#### Additional params
Define the DRUM_JAVA_XMX environment variable to set JVM maximum heap memory size (-Xmx java parameter), e.g:

```DRUM_JAVA_XMX=512m```

The DRUM tool currently supports models with DataRobot-generated Scoring Code or models that implement either the `IClassificationPredictor`
or `IRegressionPredictor` interface from [datarobot-prediction](https://mvnrepository.com/artifact/com.datarobot/datarobot-prediction).
The model artifact must have a **jar** extension.

## Define an unstructured inference model <a name="unstructured_inference_model"></a>

Inference models support unstructured mode, where input and output are not verified and can be almost anything.
This is your responsibility to verify correctness.

See [here](https://github.com/datarobot/datarobot-user-models/tree/master/model_templates/python3_unstructured)
for an example of an unstructured template.

### Data format
When working with unstructured models DRUM supports data as a text or binary file.

### Custom hooks for Python, R, and Julia models
Include any necessary hooks in a file called `custom.py` for Python models, `custom.R` for R models, or `custom.jl` for Julia models alongside your model artifacts in your model folder:

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
- in Julia `data isa String` or `data is Base.CodeUnits`


DRUM uses `Content-Type` header to determine a type to cast `data` to. Content-Type header can be provided in request or in `--content-type` CLI argument.  
`Content-Type` header format is `type/subtype;parameter`, e.g. `text/plain;charset=utf8`. Only mimetype part `text/plain`, and `charset=utf8` parameter matter for DRUM.  
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
  
## Test an inference model with DRUM locally <a name="test_inference_model_drum"></a>

Custom model runner (DRUM) is a tool that helps to assemble, test, and run custom models. 
The custom model runner folder contains its source code. 

### To install and try out DRUM locally follow these steps:

1. Clone the repository
2. Create a virtual environment: `python3 -m virtualenv <dirname for virtual environment>`
3. Activate the virtual environment: `source <dirname for virtual environment>/bin/activate`
4. cd into the repo: `cd datarobot-user-models`
5. Install the required dependencies: `pip install -r public_dropin_environments/python3_sklearn/requirements.txt`
6. Install datarobot-drum: `pip install datarobot-drum`
    1. If you want to install the dev environment, instead `pip install -e custom_model_runner/`
    
### Here are a few ways to use DRUM to test a custom model locally:
* Score data:
  * Run the example: `drum score --code-dir model_templates/python3_sklearn --input tests/testdata/juniors_3_year_stats_regression.csv`
   > Note: this command assumes model is regression.
   For binary classification model provide: _**positive-class-label**_
   and _**negative-class-label**_ arguments.  
   Input data is expected to be in CSV format. By default, missing values are indicated with NaN in Python, and NA in R according to `pd.read_csv` and `read.csv` respectively.
   
* [Test model performance](https://github.com/datarobot/datarobot-user-models/tree/master/custom_model_runner#testing-model-performance)
* [Model validation](https://github.com/datarobot/datarobot-user-models/tree/master/custom_model_runner#model-validation-checks)

## Building your own environment <a name="build_own_environment"></a>

>Note: DataRobot recommends using an environment template and not building your own environment except for specific use cases. (For example: you don't want to use DRUM but you want to implement your own prediction server.)

If you'd like to use a tool/language/framework that is not supported by our template environments, you can make your own. DataRobot recommends modifying the provided environments to suit your needs. However, to make an easy to use, re-usable environment, you should adhere to the following guidelines:

1) Your environment must include a Dockerfile that installs any requirements you may want.
2) Custom models require a simple webserver in order to make predictions. We recommend putting this in
   your environment so that you can reuse it with multiple models. The webserver must be listening on port 8080 and implement the following routes:
   > **Note: `URL_PREFIX` is an environment variable that will be available at runtime. It has to be read and pasted into the routes.**  
   > **Note: Refer to the complete API specification: [drum_server_api.yaml](custom_model_runner/drum_server_api.yaml), you can also open it rendered in the [Swagger Editor](https://editor.swagger.io/?url=https://raw.githubusercontent.com/datarobot/datarobot-user-models/master/custom_model_runner/drum_server_api.yaml).**
    1) Mandatory endpoints:
        1) `GET /URL_PREFIX/` This route is used to check if your model's server is running.
        2) `POST /URL_PREFIX/predict/` This route is used to make predictions.
    2) Nice-to-have extensions endpoints:
        1) `GET /URL_PREFIX/stats/` This route is used to fetch memory usage data for DataRobot Custom Model Testing.
        2) `GET /URL_PREFIX/health/` This route is used to check if model is loaded and functioning properly. If model loading fails error with 513 response code should be returned.  
           Failing to handle this case may cause backend k8s container to enter crash/restart loop for several minutes.
3) An executable `start_server.sh` file is required to start the model server.
4) Any code and `start_server.sh` should be copied to `/opt/code/` by your Dockerfile

See templates [here](https://github.com/datarobot/datarobot-user-models/tree/master/public_dropin_environments)
for examples

## Upload, manage and deploy a custom inference model with DataRobot <a name="upload_custom_model"></a>
Follow steps from the [docs](https://app.datarobot.com/docs/mlops/deployment/custom-models/custom-inf-model.html)