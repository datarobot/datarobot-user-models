## About
The DataRobot Model Runner (DRUM) is a tool that allows you to work locally with Python, R, and Java custom models.
It can be used to verify that a custom model can run and make predictions before it is uploaded to DataRobot.
However, this testing is only for development purposes. DataRobot recommends that any model you wish to deploy should also be tested in the Custom Model Workshop.

DRUM can also:
- run performance and memory usage testing for models.
- perform model validation tests, e.g., checking model functionality on corner cases, like null values imputation.
- run models in a Docker container.

## Communication
- open an issue in the [DRUM GitHub repository](https://github.com/datarobot/datarobot-user-models/issues).
- ask a question on the [#drum (IRC) channel](https://webchat.freenode.net/?channels=#drum).

## Custom inference models quickstart guide
View examples [here](https://github.com/datarobot/datarobot-user-models#quickstart).

## Custom training models
View examples [here](https://github.com/datarobot/datarobot-user-models#training_model_folder).

## Installation

### Prerequisites:
All models:
- Install the dependencies needed to run your code.
- If you are using a drop-in environment found in this repo, you must pip install these dependencies.

Python models:
- Python 3.6 or 3.7 is required unless you are using DRUM with a Docker image.
- This is because DRUM only runs with Python 3.6 or 3.7.

Java models:
- JRE >= 11.

R models:
- Python >= 3.6.
- The R framework must be installed.
- DRUM uses the `rpy2` package (by default the latest version is installed) to run R.
You may need to adjust the **rpy2** and **pandas** versions for compatibility.

To install DRUM with Python/Java models support:  
```pip install datarobot-drum```

To install DRUM with R support:  
```pip install datarobot-drum[R]```

### Autocompletion
DRUM supports autocompletion based on the `argcomplete` package. Additional configuration is required to use it:
- run `activate-global-python-argcomplete --user`; this should create a file: `~/.bash_completion.d/python-argcomplete`,
- source created file `source ~/.bash_completion.d/python-argcomplete` in your `~/.bashrc` or another profile-related file according to your system.

If global completion is not completing your script, bash may have registered a default completion function:
- run `complete | grep drum`; if there is an output `complete -F _minimal <some_line_containing_drum>` do
- `complete -r <some_line_containing_drum>`

For more information and troubleshooting visit the [argcomplete](https://pypi.org/project/argcomplete/) information page.


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
The *--code-dir* (code directory) argument is required in all commands and should point to a folder which contains your model artifacts and any other code needed for DRUM to run your model. For example, if you're running DRUM from **testdir** with a test input file at the root and your model in a subdirectory called **model**, you would enter:

`drum score --code-dir ./model/ --input ./testfile.csv`

#### Additional model code dependencies
Code dir may contain a `requirements.txt` file, listing dependency packages which are required by code. Only Python and R models are supported.

**Format of requirements.txt file**
* for Python: pip requrements file format
* for R: a package per line

DRUM will attempt to install dependencies only when running with [`--docker`](#docker) option.

### Model template generation
<a name="new"></a>
DRUM can help you to generate a code folder template with the `custom` file described above.  
`drum new model --code-dir ~/user_code_dir/ --language r`  
This command creates a folder with a `custom.py/R` file and a short description: `README.md`.

### Batch scoring mode
<a name="score"></a>
> Note: DRUM doesn't automatically distinguish between regression and classification. When you are using classification model, provide: _**positive-class-label**_ and _**negative-class-label**_ arguments.
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
For more feature options, see:
```drum perf-test --help```

### Model validation checks
<a name="validation"></a>
You can validate the model on a set of various checks.
It is highly recommended to run these checks, as they are performed in DataRobot before the model can be deployed.

List of checks:
- null values imputation: each feature of the provided dataset is set to missing and fed to the model.

To run:
```drum validation --code-dir ~/user_code_dir/ --input 10k.csv --positive-class-label yes --negative-class-label no```
Sample report:
```
Validation check results
Test case         Status
==============================
Null value imputation   PASSED
```
In case of check failure more information will be provided.


### Prediction server mode
<a name="server"></a>

DRUM can also run as a prediction server. To do so, provide a server address argument:  
```drum server --code-dir ~/user_code_dir --address localhost:6789```

The DRUM prediction server provides the following routes. You may provide the environment variable URL_PREFIX. Note that URLs must end with /.

* Status routes:   
A GET **URL_PREFIX/** and **URL_PREFIX/ping/** routes, shows server status - if the server is alive.  
Example: GET http://localhost:6789/

* Health route:  
A GET **URL_PREFIX/health/** route, shows functional health. E.g. model is loaded and functioning properly.  
Example: GET http://localhost:6789/
  
* Info route:  
A GET **URL_PREFIX/info/** route, shows information about running model (metadata, paths, predictor type, etc.).  
Example: GET http://localhost:6789/

* Statistics route:  
A GET **URL_PREFIX/stats/** route, shows running model statistics (memory).  
Example: GET http://localhost:6789/

* Capabilities route:  
A GET **URL_PREFIX/capabilities/** route, shows payload formats supported by running model.  
Example: GET http://localhost:6789/

* Shutdown route:   
A POST **URL_PREFIX/shutdown/** route, shuts the server down.  
Example: POST http://localhost:6789/shutdown/

* Structured predictions routes:   
A POST **URL_PREFIX/predict/** and **URL_PREFIX/predictions/** routes, which returns predictions on data.  
Example: POST http://localhost:6789/predict/; POST http://localhost:6789/predictions/  
For these routes data can be posted in two ways:
  * as form data parameter with a <key:value> pair, where:  
key = X  
value = filename of the `csv/arrow/mtx` format, that contains the inference data.
  * as binary data; in case of `arrow` or `mtx` formats, mimetype `application/x-apache-arrow-stream` or `text/mtx` must be set.
   
* Structured transform route (for Python predictor only):   
A POST **URL_PREFIX/transform/** route, which returns transformed data.  
Example: POST http://localhost:6789/transfor/;  
For this route data can be posted in two ways:
  * as form data parameter with a <key:value> pair, where:  
key = `X`.  
value = filename of the `csv/arrow/mtx` format, that contains the inference data.
 
    optionally a second key, `y`, can be passed with value = a second filename containing target data. 
    
    if `y` is passed, the route will return both `X.transformed` and `y.transformed` keys, along with `out.format`
     indicating the format of the transformed X output. This will take a value of `csv`, 
    `sparse` or `arrow`. `y.transformed` is never sparse.
    
    an `arrow_version` key may also be passed if you desire to use `arrow` format for `X.transformed` or `y.transformed`.
    this is used to ensure that the endpoint returns data that can be opened by the caller's version of arrow. without this
    key, all dense data returned will default to csv format.

  * as binary data; in case of `arrow` or `mtx` formats, mimetype `application/x-apache-arrow-stream` or `text/mtx` must be set.
  
  
* Unstructured predictions routes:  
A POST **URL_PREFIX/predictUnstructured/** and **URL_PREFIX/predictionsUnstructured/** routes, which returns predictions on data.  
Example: POST http://localhost:6789/predictUnstructured/; POST http://localhost:6789/predictionsUnstructured/  
For these routes data is posted as binary data. Provide mimetype and charset to properly handle the data.
For more detailed information please go [here](https://github.com/datarobot/datarobot-user-models#unstructured_inference_models).  

#### Starting drum as prediction server in production mode.
Drum prediction server can be started in *production* mode which has nginx and uwsgi as the backend.
This provides better stability and scalability - depending on how many CPUs are available several workers will be started to serve predictions.  
*--max-workers* parameter  can be used to limit number of workers.  
E.g. ```drum server --code-dir ~/user_code_dir --address localhost:6789 --production --max-workers 2```

> Note: *Production* mode may not be available on Windows-based systems out ot the box, as uwsgi installation requires special handling.
> Docker container based Linux environment can be used for such cases.

### Fit mode
<a name="fit"></a>
> Note: Running fit inside of DataRobot is currently in alpha. Check back soon for the opportunity
to test out this functionality yourself.

DRUM can run your training model to make sure it can produce a trained model artifact before
adding the training model into DataRobot.

You can try this out on our sklearn classifier model template this this command:

```
drum fit --code-dir model_templates/python3_sklearn_binary --target-type binary --target Species --input \
tests/testdata/iris_binary_training.csv --output . --positive-class-label Iris-setosa \
--negative-class-label Iris-versicolor
```
> Note: If you don't provide class label, DataRobot tries to autodetect the labels for you.

You can also use DRUM on regression datasets, and soon you will also be able to provide row weights. Checkout the ```drum fit --help``` output for further details.



### Running inside a docker container
<a name="docker"></a>
In every mode, DRUM can be run inside a docker container by providing the option ```--docker <image_name/directory_path>```.
The container should implement an environment required to perform desired action.
DRUM must be installed as a part of this environment.  
The following is an example on how to run DRUM inside of container:  
```drum score --code-dir ~/user_code_dir/ --input dataset.csv --docker <container_name>```  
```drum perf-test --code-dir ~/user_code_dir/ --input dataset.csv --docker <container_name>```

Alternatively, the argument passed through the `--docker` flag may be a directory containing the unbuilt contents
of an image. The DRUM tool will then attempt to build an image using this directory and run your model inside
the newly built image.

If the argument passed to `--docker` is a docker context directory, and code dir contains dependencies file `requirements.txt`, DRUM will try to install the packages during the image build.  
To skip dependencies installation you can use `--skip-deps-install` flag. 

## Drum Push
Starting in version 1.1.4, drum includes a new verb called `push`. When the user writes
`drum push -cd /dirtopush/` the contents of that directory will be submitted as a custom model
to DataRobot. However, for this to work, you must create two types of configuration.
1. **DataRobot client configuration**
`push` relies on correct global configuration of the client to access a DataRobot server.
There are two options for supplying this configuration, through environment variables or through
a config file which is read by the DataRobot client. Both of these options will include an endpoint
and an API token to authenticate the requests.

* Option 1: Environment variables.
    Example:
    ```
    export DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2
    export DATAROBOT_API_TOKEN=<yourtoken>
    ```
* Option 2: Create this file, which we check for: `~/.config/datarobot/drconfig.yaml`  
    Example:
    ```
    endpoint: https://app.datarobot.com/api/v2
    token: <yourtoken>
    ```
2. **Model Metadata** `push` also relies on a metadata file, which is parsed on DRUM to create
the correct sort of model in DataRobot. This metadata file includes quite a few options. You can
[read about those options](https://github.com/datarobot/datarobot-user-models/blob/master/MODEL-METADATA.md) or [see an example](https://github.com/datarobot/datarobot-user-models/blob/master/model_templates/inference/python3_sklearn/model-metadata.yaml).
