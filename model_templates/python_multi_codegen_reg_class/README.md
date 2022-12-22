# Python + Multiple CodeGen models

Current example implements a Python model that loads multiple Scoring Code models: binary/multiclass classification,
regression.   
For the binary and regression Scoring Code models predictions explanations are supported.

# Prerequisites
- This example is a part of PoC, so DRUM 1.9.9.dev2 version is required to run the example.
- Java 11



# Contents overview
All the models are trained on the `./tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv` dataset.
The `class` feature has three values: STAR, GALAXY, QSO (For the binary classification case QSO values has been removed)

The model folder contains:
- `skyserver_binary_star_galaxy` - folder with binary classification Scoring Code artifact with prediction explanations.
- `skyserver_multiclass` - folder with multi classification Scoring Code artifact WITHOUT prediction explanations (not supported for Scoring Code multi classification).
- `skyserver_reg` - folder with regression Scoring Code artifact with prediction explanations.
- `custom.py` - model custom code.
- `requirements.txt` - additional packages required to run the model. It will take effect once model is assembled in the DR app.

# `custom.py` overview
## class ScoringCodePredictor
Class that loads a Scoring Code artifact and provides APIs to make predictions. 
In the example artifacts are loaded with parameters appropriate to every target type.

- if model is loaded with `with_explanations=True` `predict_with_explanations()` API should be used;
- if with `with_explanations=False` - `predict()` API.

Return values are:
- ```predictions_df = predict()```
- ```predictions_df, explanations_df = predict_with_explanations()```

## `load_model` hook
`load_model` hook must implement model loading and return a single object, which will be passed as a model parameter into the other hooks.

## `score` hook
`score` implements predictions and should return dataframe appropriate to the target-type specified in the `drum` call or defined 
during custom model creation in the DR app.

## `score_unstructured` hook
`score_unstructured` implements predictions and should return generic str/bytes object, assembled by user.


# Running locally
## Build environment
The easiest way to run locally is to build an environment docker.

Paths are relative to the repository root - `./datarobot-user-models`:    
- `cd public_dropin_environments/python3_sklearn`
- Edit `dr_requirements.txt` file to list `datarobot-drum==1.9.9.dev2`
- Build environment: `docker build -t python3_sklearn .`

## Structured multiclass example
### Run DRUM in batch mode
Paths are relative to the repository root - `./datarobot-user-models`:  
`drum score --code-dir ./model_templates/python_multi_codegen_reg_class/ --target-type multiclass --input tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv --class-labels STAR GALAXY QSO --docker python3_sklearn --verbose`

`score` hook must return multiclass predictions as `--target-type multiclass` is specified

### Run DRUM as a server
Paths are relative to the repository root - `datarobot-user-models`:  
`drum server --code-dir ./model_templates/python_multi_codegen_reg_class/ --target-type multiclass --class-labels STAR GALAXY QSO --docker python3_sklearn --address 0.0.0.0:4567`  

To make predictions:  
`curl -X POST --form "X=@./tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv" 0.0.0.0:4567/predict/`  
or  
`curl -X POST -H "Content-Type: text/plain" --data-binary "@./tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv" 0.0.0.0:4567/predict/`


## Unstructured example with prediction explanations
### Run DRUM in batch mode
Paths are relative to the repository root - `./datarobot-user-models`:  
`drum score --code-dir ./model_templates/python_multi_codegen_reg_class/  --target-type unstructured --input tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv --docker python3_sklearn --verbose`

`score_unstructured` hook must return generic txt/bytes.
In the current example predictions and explanations are assembled together and returned as a JSON string.

### Run DRUM as a server
Paths are relative to the repository root - `./datarobot-user-models`:  
`drum server --code-dir ./model_templates/python_multi_codegen_reg_class/ --target-type unstructured --docker python3_sklearn --address 0.0.0.0:4567`  

To make predictions:  
`curl -X POST -H "Content-Type: text/plain" --data-binary "@./tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv" 0.0.0.0:4567/predictUnstructured/`
