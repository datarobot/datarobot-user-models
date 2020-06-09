# README

DataRobot User Model runner - **drum**

## About
The DataRobot Model Runner - **drum** - is a tool that allows you to work locally with Python, R, and Java DataRobot custom models.
It can be used to verify that a custom model can run and make predictions before it is uploaded to the DataRobot.
However, this testing is only for development purposes. DataRobot recommends that any model you wish to deploy should also be tested in the Custom Model Workshop after uploading it.
  
**drum** can also:
- run performance and memory usage testing for models,
- perform model validation tests - check model functionality on corner cases, like null values imputation,
- run models in a docker container.

## Installation

### Prerequisites:
All models:
- It is on you to make sure that you install the dependencies needed to run your code. 
- If you are using a Dropin Environment found in this repo, you will have to pip install these dependencies yourself.

Python models:
- Python 3 is required, unless you are using drum with a docker image. This is because the DRUM tool only runs with python 3 itself.
  
Java models:
- JRE >= 11.

R models:
- Python >= 3.6.
- R framework installed.
- **drum** uses `rpy2` package (by default the latest version is installed) to run R.
You may need to adjust **rpy2** and **pandas** versions for compatibility.

Install **drum** with Python/Java models support:  
```pip install datarobot-drum```

Install **drum** with R support:  
```pip install datarobot-drum[R]```

### Running examples
Clone the DataRobot User Models repo from: https://github.com/datarobot/datarobot-user-model
- change dir `cd datarobot-user-model`
- create virtual environment `python3 -m venv ~/drum-virt-env`
- activate virtual environment `. ~/drum-virt-env/bin/activate`
- install example dependencies `pip install -r requirements.txt`
- install **drum** `pip install datarobot-drum`
- go to the directory with model templates `cd model_templates`

Check README file in every directory for the exact command to run the example.
Copy/paste command to run it from the current path `datarobot-user-models/model_templates` e.g.:  
`drum score --code-dir ./python3_sklearn_inference --input ../tests/testdata/boston_housing.csv`



### Autocompletion
**drum** supports autocompletion based on the `argcomplete` package. Additional configuration is required to use it:
- run `activate-global-python-argcomplete --user`; this should create a file: `~/.bash_completion.d/python-argcomplete`,
- source created file `source ~/.bash_completion.d/python-argcomplete` in your `~/.bashrc` or another profile-related file according to your system.

If global completion is not completing your script, bash may have registered a default completion function:
- run `complete | grep drum`; if there is an output `complete -F _minimal <some_line_containing_drum>` do
- `complete -r <some_line_containing_drum>`

For more information and troubleshooting visit the [argcomplete](https://pypi.org/project/argcomplete/) information page.

## Built-In Model Support
**drum** has built in support for the following libraries; if your model is based on one of these libraries, **drum** expects your model artifact to have a matching file extension.

### Python
| Library | File Extension | Example |
| --- | --- | --- |
| scikit-learn | *.pkl | sklean-regressor.pkl |
| xgboost | *.pkl | xgboost-regressor.pkl |
| PyTorch | *.pth | torch-regressor.pth |
| keras | *.h5 | keras-regressor.h5 |

### R
| Library | File Extension | Example |
| --- | --- | --- |
| caret | *.rds | brnn-regressor.rds |

This tool makes the following assumption about your serialized model:
- The data sent to a model can be used to make predictions without
additional pre-processing
- Regression models return a single floating point per row of prediction data
- Binary classification models return two floating point values that sum to 1.0 per row of prediction data
  - The first value is the positive class probability, the second is the negative class probability
- There is a single pkl/pth/h5 file present
- Your model uses one of the above frameworks
  
### Custom hooks for Python and R models
If the assumptions mentioned above are incorrect for your model, **drum** supports several hooks for custom code. If needed,
include any necessary hooks in a file called `custom.py` for python models or `custom.R` for R models alongside your model artifacts in your model folder:

> _**NOTE:**_ The following hook signatures are written with Python 3 type annotations. The Python types match the following R types
> - DataFrame = data.frame
> - None = NULL
> - str = character
> - Any = R Object (the deserialized model)
> - *args, **kwargs = ... (these aren't types, they're just placeholders for additional parameter)

- `init(**kwargs) -> None`
  - Executed once in the beginning of the run.
  - `kwargs` - additional keyword arguments to the method;
    - code_dir - code folder passed in the `--code_dir` parameter
- `load_model(code_dir: str) -> Any`
  - `code_dir` is the directory where model artifact and additional code are provided, passed in the `--code_dir` parameter
  - If used, this hook must return a non-None value
  - Can be used to load supported models if your model has multiple artifacts, or for loading models that
  **drum** does not natively support
- `transform(data: DataFrame, model: Any) -> DataFrame`
  - `data` is the dataframe given to **drum** to make predictions on
  - `model` is the deserialized model loaded by **drum** or by `load_model`, if supplied
  - Intended to apply transformations to the prediction data before making predictions. This is most useful
  if **drum** supports the model's library, but your model requires additional data processing before it can make predictions
- `score(data: DataFrame, model: Any, **kwargs: Dict[str, Any]) -> DataFrame`
  - `data` is the dataframe to make predictions against. If `transform` is supplied, `data` will be the transformed data.
  - `model` is the deserialized model loaded by **drum** or by `load_model`, if supplied
  - `kwargs` - additional keyword arguments to the method;  
    In case of classification model class labels will be provided as the following arguments:
    - `positive_class_label` is the positive class label for a binary classification model
    - `negative_class_label` is the negative class label for a binary classification model
  - This method should return predictions as a dataframe with the following format:
    - Binary Classification: must have columns for each class label with floating- point class probabilities as values. Each row
    should sum to 1.0
    - Regression: must have a single column called `Predictions` with numerical values
  - This hook is only needed if you would like to use **drum** with a framework not natively supported by the tool.
- `post_process(predictions: DataFrame, model: Any) -> DataFrame`
  - `predictions` is the dataframe of predictions produced by **drum** or by the `score` hook, if supplied
  - `model` is the deserialized model loaded by **drum** or by `load_model`, if supplied
  - This method should return predictions as a dataframe with the following format:
    - Binary Classification: must have columns for each class label with floating- point class probabilities as values. Each row
    should sum to 1.0
    - Regression: must have a single column called `Predictions` with numerical values
  - This method is only needed if your model's output does not match the above expectations
> *Note: training and inference hooks can be defined in the same file*
### Java
| Library | File Extension | Example |
| --- | --- | --- |
| datarobot-prediction | *.jar | dr-regressor.jar |

**drum** currently supports models with DataRobot-generated Scoring Code or models that implement the either the `IClassificationPredictor`
or `IRegressionPredictor` interface from the [datarobot-prediction](https://mvnrepository.com/artifact/com.datarobot/datarobot-prediction).
The model artifact must have a **jar** extension.

#### Additional params
Define DRUM_JAVA_XMX environment variable to set JVM maximum heap memory size (-Xmx java parameter), e.g:
```DRUM_JAVA_XMX=512m```

## Training models. Content of the model folder
The model folder must contain any code needed for **drum** to run to train your model.

### Python
Model folder must contain a `custom.py` file which defines a `fit` method.

- `fit(X: pandas.DataFrame, y: pandas.Series, output_dir: str, **kwargs: Dict[str, Any]) -> None`
    - `X` is the dataframe to perform fit on.
    - `y` is the dataframe containing target data.
    - `output_dir` is the path to write model artifact to.
    - `kwargs` additional keyword arguments to the method;
        - `class_order: List[str]` a two element long list dictating the order of classes which should be used for modeling.
        - `row_weights: np.ndarray` an array of non-negative numeric values which can be used to dictate how important a row is.

> *Note: training and inference hooks can be defined in the same file*

## Usage
Help:  
```**drum** -help```

### Operations
- [score](#score)
- [fit](#fit)
- [perf-test](#perf)
- [validation](#validation)
- [server](#server)
- [new](#new)


### Code Directory *--code-dir*
The *--code-dir (code directory)* argument is required in all commands and should point to a folder which contains your model artifacts and any other code needed for **drum** to run your model. For example if running **drum** from **testdir** with a test input file at the root and your model in a subdirectory called **model** you would enter

`drum score --code-dir ./model/ --input ./testfile.csv`

### Model template generation
<a name="new"></a>
drum can help you to generate a code folder template with the `custom` file described above.  
`drum new model --code-dir ~/user_code_dir/ --language r`  
This command creates a folder with a `custom.py/R` file and a short description - `README.md`.

### Batch scoring mode
<a name="score"></a>
#### Run a binary classification custom model
Make batch predictions with a binary classification model. Optionally, specify an output file. Otherwise, predictions are returned to the command line:  
```drum score --code-dir ~/user_code_dir/ --input 10k.csv  --positive-class-label yes --negative-class-label no --output 10k-results.csv --verbose```

#### Run a regression custom model
Make batch predictions with a regression model:  
```drum score --code-dir ~/user_code_dir/ --input fast-iron.csv --verbose```

### Testing model performance
<a name="perf"></a>
You can test how the model performs and get its latency times and memory usage.  
In this mode, the model is started with a prediction server. Different request combinations are submitted to it.
After it completes, it returns a report.  
```drum perf-test --code-dir ~/user_code_dir/ --input 10k.csv --positive-class-label yes --negative-class-label no```  
Report example:
```
samples   iters    min     avg     max    used (MB)   total (MB)
============================================================================
Test case         1     100   0.028   0.030   0.054     306.934    31442.840
Test case        10     100   0.030   0.034   0.069     307.375    31442.840
Test case       100      10   0.036   0.038   0.045     307.512    31442.840
Test case      1000      10   0.042   0.047   0.058     308.258    31442.840
Test case    100000       1   0.674   0.674   0.674     330.902    31442.840
50MB file    838861       1   5.206   5.206   5.206     453.121    31442.840
```
For more feature options see:
```drum perf-test --help```

### Model validation checks
<a name="validation"></a>
You can validate the model on a set of various checks.
It is highly recommended to run these checks, as they are performed in the DataRobot app before model can be deployed.
   
List of checks:
- null values imputation: each feature of the provided dataset is set to missing and fed to the model.

To run:
```drum validation --code-dir ~/user_code_dir/ --input 10k.csv --positive-class-label yes --negative-class-label no```
Report example:
```
Validation check results
Test case         Status
==============================
Null value imputation   PASSED
```
In case of check failure more information will be provided.


### Prediction server mode
<a name="server"></a>

The **drum** can run as a prediction server. To do so, provide a server address argument:  
```drum server --code-dir ~/user_code_dir --address localhost:6789```

The **drum** prediction server provides the following routes. You may provide the environment variable URL_PREFIX. Note that URLs must end with /.

* A GET URL_PREFIX/ route, which checks if the server is alive.  
Example: GET http://localhost:6789/

* A POST URL_PREFIX/shutdown/ route, which shuts the server down.  
Example: POST http://localhost:6789/shutdown/

* A POST URL_PREFIX/predict/ route, which returns predictions on data.  
Example: POST http://localhost:6789/predict/  
For this /predict/ route, provide inference data (for the model to make predictions) as form data with a <key:value> pair, where:  
key = X  
value = filename of the CSV that contains the inference data

### Fit mode
<a name="fit"></a>
NOTE: Currently, running fit inside of DataRobot is in alpha. Check back soon for the opportunity
to test out this functionality for yourself.

The **drum** can run your training model to make sure it can produce a trained model artifact before
adding the training model into DataRobot. 

You can try this out on our sklearn classifier model template this with the command 

```
drum fit --code-dir model_templates/python3_sklearn --target Species --input \
tests/testdata/iris_binary_training.csv --output . --positive-class-label Iris-setosa \
--negative-class-label Iris-versicolor
```
You can also use **drum** on regression datasets, and soon, you will be able to provide row weights
as well. Checkout the ```drum fit --help``` output for further details. 



### Running inside a docker container
In every mode **drum** can be run inside a docker container by providing an option ```--docker <image_name>```.
The container should implement an environment required to preform desired action.
**drum** must be installed as a part of this environment.  
Example on how to run inside of container:  
```drum score --code-dir ~/user_code_dir/ --input dataset.csv --docker <container_name>```  
```drum perf-test --code-dir ~/user_code_dir/ --input dataset.csv --docker <container_name>```
