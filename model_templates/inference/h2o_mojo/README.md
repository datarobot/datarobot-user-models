## H2O Mojo Template

These models are intended to work with the [Java H2O Drop-In Environment](../../public_dropin_environments/java_h2o/).

The models provided in each folder are H2O models exported as MOJOs.  For more details see [H2O MOJO](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/productionizing.html)


### Instructions

Upload H2O Mojo file (Modern Object, Optimized) and use the Java H2O Drop-In Environment with it.  Mojo file has a `.zip` extension.  No others files are necessary

### Examples

* Binary - The binary example is based on the iris dataset with target `Species`
* regression - boston housing pricing dataset with target `MEDV`. 

### To run locally using 'drum'

To run these examples locally with `drum` installed, you must already have java 11 installed, or you can execute the examples with Docker.  

Paths are relative to `./datarobot-user-models/model_templates/inference` unless fully qualified

#### Binary 

`drum score --code-dir ./h2o_mojo/binary --input ../../tests/testdata/iris_binary_training.csv --positive-class-label 1 --negative-class-label 0` 

#### Regression 

`drum score --code-dir ./h2o_mojo/regression --input ../../tests/testdata/boston_housing.csv`

#### Docker

If you do not have Java 11 installed, please consider using docker.  

You can either provide the path to the Dockerfile

`drum score --code-dir ./h2o_mojo/binary --input ../../tests/testdata/iris_binary_training.csv --positive-class-label 1 --negative-class-label 0 --docker ../../public_dropin_environments/java_h2o/`

or provide the name of the docker image that has already been built [Java H2O Drop-In Environment](../../public_dropin_environments/java_h2o/), for example, docker image is `drum_h2o`.

`drum score --code-dir ./h2o_mojo/binary --input ../../tests/testdata/iris_binary_training.csv --positive-class-label 1 --negative-class-label 0 --docker drum_h2o`
