## H2O POJO Template

__WIP__
__TESTS ARE COMING__

These models are inteded to work with the [Java Drop-In Environment](../../../public_dropin_environments/java_codegen/).

The models provided in each folder are H2O models exported as POJO.  For more details see [H2O POJO](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/productionizing.html)

├── binary
│   └── XGBoost_grid__1_AutoML_20200717_163214_model_159.java
└── regression
    └── drf_887c2e5b_0941_40b7_ae26_cae274c4b424.java


### Instructions

For the time being, the POJO must be compiled and the entire folder would by loaded and used with Java Drop-In Environment with it. 

### Examples

* Binary - The binary example is based on the iris dataset
* regression - boston housing pricing dataset with target `MEDV`. 

### To run locally using 'drum'
Paths are relative to `./datarobot-user-models/model_templates/inference/h2o_pojo`:  

#### Binary 

First, compile the pojo

`javac -cp h2o-genmodel-3.30.0.6.jar binary/XGBoost_grid__1_AutoML_20200717_163214_model_159.java`

`drum score --code-dir ./binary --input ../../../tests/testdata/iris_binary_training.csv --positive-class-label 1 --negative-class-label 0`

`drum score --code-dir ./h2o_pojo/binary --input ../../tests/testdata/iris_binary_training.csv --positive-class-label 1 --negative-class-label 0 --docker drum_h2o`


#### Regression 

First, compile the pojo

`javac -cp h2o-genmodel-3.30.0.6.jar regression/drf_887c2e5b_0941_40b7_ae26_cae274c4b424.java`

`drum score --code-dir ./regression --input ../../../tests/testdata/boston_housing.csv`

`drum score --code-dir ./h2o_pojo/regression --input ../../tests/testdata/boston_housing.csv --docker drum_h2o`

drum server --code-dir ./h2o_pojo/regression --address localhost:6789 --docker drum_h2o
