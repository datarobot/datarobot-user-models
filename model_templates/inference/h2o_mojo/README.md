## H2O Mojo Template

__WIP__
__TESTS ARE COMING__

These models are intended to work with the [Java H2O Drop-In Environment](../../../public_dropin_environments/java_h2o/).

The models provided in each folder are H2O models exported as MOJOs.  For more details see [H2O MOJO](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/productionizing.html)

├── binary
│   └── XGBoost_grid__1_AutoML_20200717_163214_model_159.zip
└── regression
    └── drf_887c2e5b_0941_40b7_ae26_cae274c4b424.zip


### Instructions
Upload the a H2O Mojo file as the only file in the custom model and use the Java Drop-In Environment with it. 

### Examples

* Binary - The binary example is based on the iris dataset
* regression - boston housing pricing dataset with target `MEDV`. 

### To run locally using 'drum'
Paths are relative to `./datarobot-user-models/model_templates/inference`:  

#### Binary 
`drum score --code-dir ./h2o_mojo/binary --input ../../tests/testdata/iris_binary_training.csv --positive-class-label 1 --negative-class-label 0 --docker drum_h2o`

#### Regression 
`drum score --code-dir ./h2o_mojo/regression --input ../../tests/testdata/boston_housing.csv --docker drum_h2o`

#### Server
`drum server --code-dir ./h2o_mojo/regression --address localhost:6789 --docker drum_h2o_v2 --docker drum_h2o`
