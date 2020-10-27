
## H2O Driverless AI MOJO Pipeline

These models are inteded to work with the [Java H2O Drop-In Environment](../../public_dropin_environments/java_h2o/).

 No examples are included.  Users must a have a valid ongoing Driverless AI license to use exported pipelines.  We provide no license and do not include this framework in our automated testing.  

The Driverless AI Pipeline internals are used to provide the class names for the classification models. So while you must provide them on the command line utilities, they will not be passed through to the model.

Please see the [documentation](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/python-mojo-pipelines.html) for additional details.  


### Instructions

Unzip the exported DAI Pipeline.  You can place your `license.sig` file in the unzip mojo, or set environment variables appropriately.  The environment variables youmust set are either one of the following 

* `DRIVERLESS_AI_LICENSE_FILE` : A location of file with a license
* `DRIVERLESS_AI_LICENSE_KEY` : A license key

### Examples

NO EXAMPLES PROVIDED

### To run locally using 'drum'

To run these examples locally with `drum` installed, you must already have java 11 installed, or you can execute the examples with Docker.  

Paths are relative to `./datarobot-user-models/model_templates/inference` unless fully qualified

All folders listed below in the `--code-dir` arguemtns are assumed to be unzipped DAI Mojo Pipelines, and users are expected to have a valid ongoing Driverless AI License. 

#### Binary 

`drum score --code-dir ./h2o_mojo_pipeline/binary --target-type binary --input ../../tests/testdata/iris_binary_training.csv --positive-class-label 1 --negative-class-label 0`

#### Multiclass 

`drum score --code-dir ./h2o_mojo_pipeline/multiclass --target-type multiclass --input ../../tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv` 

#### Regression 

`drum score --code-dir ./h2o_mojo_pipeline/regression --target-type regression --input ../../tests/testdata/boston_housing.csv`

#### Docker

If you do not have Java 11 installed, please consider using docker.  

You can either provide the path to the Dockerfile

`drum score --code-dir ./h2o_mojo_pipeline/binary --target-type binary --input ../../tests/testdata/iris_binary_training.csv --positive-class-label 1 --negative-class-label 0 --docker ../../public_dropin_environments/java_h2o/`

or provide the name of the docker image that has already been built [Java H2O Drop-In Environment](../../public_dropin_environments/java_h2o/), for example, docker image is `drum_h2o`.

`drum score --code-dir ./h2o_mojo_pipeline/binary --target-type binary --input ../../tests/testdata/iris_binary_training.csv --positive-class-label 1 --negative-class-label 0 --docker drum_h2o`
