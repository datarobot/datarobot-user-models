# README

This page provides an overview of the DataRobot User Model runner (**drum**).

## About
The DataRobot Model Runner (**drum**) is a tool that allows you to work locally with Python, R, and Java custom models.
It can be used to verify that a custom model can run and make predictions before it is uploaded to DataRobot.
However, this testing is only for development purposes. DataRobot recommends that any model you wish to deploy should also be tested in the Custom Model Workshop.
  
**drum** can also:
- run performance and memory usage testing for models.
- perform model validation tests, e.g., checking model functionality on corner cases, like null values imputation.
- run models in a Docker container.

## Custom inference models
View examples [here](https://github.com/datarobot/datarobot-user-models#quickstart).

## Custom training models
View examples [here](https://github.com/datarobot/datarobot-user-models#training_model_folder).

## Installation

### Prerequisites:
All models:
- Install the dependencies needed to run your code. 
- If you are using a drop-in environment found in this repo, you must pip install these dependencies.

Python models:
- Python 3 is required unless you are using **drum** with a Docker image. This is because **drum** only runs with Python 3.
  
Java models:
- JRE >= 11.

R models:
- Python >= 3.6.
- The R framework must be installed.
- **drum** uses the `rpy2` package (by default the latest version is installed) to run R.
You may need to adjust the **rpy2** and **pandas** versions for compatibility.

To install **drum** with Python/Java models support:  
```pip install datarobot-drum```

To install **drum** with R support:  
```pip install datarobot-drum[R]```

### Autocompletion
**drum** supports autocompletion based on the `argcomplete` package. Additional configuration is required to use it:
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
The *--code-dir* (code directory) argument is required in all commands and should point to a folder which contains your model artifacts and any other code needed for **drum** to run your model. For example, if you're running **drum** from **testdir** with a test input file at the root and your model in a subdirectory called **model**, you would enter:

`drum score --code-dir ./model/ --input ./testfile.csv`

### Model template generation
<a name="new"></a>
**drum** can help you to generate a code folder template with the `custom` file described above.  
`drum new model --code-dir ~/user_code_dir/ --language r`  
This command creates a folder with a `custom.py/R` file and a short description: `README.md`.

### Batch scoring mode
<a name="score"></a>
> Note: **drum** doesn't automatically distinguish between regression and classification. When you are using classification model, provide: _**positive-class-label**_ and _**negative-class-label**_ arguments.
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

**drum** can also run as a prediction server. To do so, provide a server address argument:  
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
> Note: Running fit inside of DataRobot is currently in alpha. Check back soon for the opportunity
to test out this functionality yourself.

**drum** can run your training model to make sure it can produce a trained model artifact before
adding the training model into DataRobot. 

You can try this out on our sklearn classifier model template this this command: 

```
drum fit --code-dir model_templates/python3_sklearn --target Species --input \
tests/testdata/iris_binary_training.csv --output . --positive-class-label Iris-setosa \
--negative-class-label Iris-versicolor
```
> Note: If you don't provide class label, DataRobot tries to autodetect the labels for you. 

You can also use **drum** on regression datasets, and soon you will also be able to provide row weights. Checkout the ```drum fit --help``` output for further details. 



### Running inside a docker container
In every mode, **drum** can be run inside a docker container by providing the option ```--docker <image_name>```.
The container should implement an environment required to perform desired action.
**drum** must be installed as a part of this environment.  
The following is an example gn how to run **drum** inside of container:  
```drum score --code-dir ~/user_code_dir/ --input dataset.csv --docker <container_name>```  
```drum perf-test --code-dir ~/user_code_dir/ --input dataset.csv --docker <container_name>```
