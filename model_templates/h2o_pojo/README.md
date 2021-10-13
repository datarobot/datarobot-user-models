## H2O POJO Template

These models are inteded to work with the [Java Drop-In Environment](../../public_dropin_environments/java_codegen/).

The models provided in each folder are H2O models exported as POJO (Plain Old Java Objects).  For more details see [H2O POJO](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/productionizing.html)

├── binary
│   └── XGBoost_grid__1_AutoML_20200717_163214_model_159.java
└── regression
    └── drf_887c2e5b_0941_40b7_ae26_cae274c4b424.java

The H2O Model internals are used to provide the class names for the classification models. So while you must provide them on the command line utilities, they will not be passed through to the model.


### Instructions

Upload H2O Pojo File and use with the [Java Drop-In Environment](../../public_dropin_environments/java_codegen/).  There is no need to compile the POJO as this be handled by drum.  Only provide one POJO.  More than one will certainly toss an exception.  

### Examples

* Binary - The binary example is based on the iris dataset `Species`
* Multiclass - multiclass example based on the galaxy dataset with target `class`
* regression - grade dataset with target `Grade 2014`. 

### To run locally using 'drum'

To run these examples locally with `drum` installed, you must already have java 11 installed, or you can execute the examples with Docker.  

Paths are relative to `./datarobot-user-models/model_templates` unless fully qualified

#### Binary 

`drum score --code-dir ./h2o_pojo/binary --target-type binary --input ../tests/testdata/iris_binary_training.csv --positive-class-label Iris-setosa --negative-class-label Iris-versicolor`

#### Multiclass 

`drum score --code-dir ./h2o_pojo/multiclass --target-type multiclass --class-labels GALAXY QSO STAR --input ../tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv` 

#### Regression 

`drum score --code-dir ./h2o_pojo/regression --target-type regression --input ../tests/testdata/juniors_3_year_stats_regression.csv --docker drum_h2o`

#### Docker

If you do not have Java 11 installed, please consider using docker.  

You can either provide the path to the Dockerfile

`drum score --code-dir ./h2o_pojo/binary --target-type binary --input ../tests/testdata/iris_binary_training.csv --positive-class-label Iris-setosa --negative-class-label Iris-versicolor --docker ../../public_dropin_environments/java_codegen/`

or provide the name of the docker image that has already been built [Java Drop-In Environment](../../public_dropin_environments/java_codegen/), for example, docker image is `drum_h2o`.

`drum score --code-dir ./h2o_pojo/binary --target-type binary --input ../tests/testdata/iris_binary_training.csv --positive-class-label Iris-setosa --negative-class-label Iris-versicolor --docker drum_h2o`
