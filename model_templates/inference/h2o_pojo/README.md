## H2O POJO Examples

These example model are intedned to work with the [Java Drop-In Environment](../../public_dropin_environments/java_codegen/).  Should be considered a __WORK IN PROGRESS__.  

The folder would contain the POJO which has extension `.java` and should be compiled locally prior to drop-in.  

for binary
`javac -cp h2o-genmodel-3.30.0.6.jar binary/XGBoost_grid__1_AutoML_20200717_163214_model_159.java`
for lending_club
`javac -cp h2o-genmodel-3.30.0.6.jar lending_club/GBM_model_python_1589382591366_1.java`
for regression 
`javac -cp h2o-genmodel-3.30.0.6.jar regression/drf_887c2e5b_0941_40b7_ae26_cae274c4b424.java`
for sar
`javac -cp h2o-genmodel-3.30.0.6.jar sar/XGBoost_model_python_1589382591366_94.java`

I think the goal is to have the class compiled when the environment is spun up.  

## Instructions
Upload the jar file as the only file in the custom model and use the Java Drop-In Environment with it

### To run locally using 'drum'
Paths are relative to `./datarobot-user-models/model_templates`:  
`drum score --code-dir ./java_codegen --input ../tests/testdata/boston_housing.csv`


Model Templates which can be executed with `drum`

## H2O Pojo Templates

With the pojo templates, you will need to compile the pojo with java 11.  

To compile, leverage `h2o-genmodel-3.30.0.6.jar` included in this directory by running the following 
from command line.  For example, the h2o_pojo_sar temp, compile the POJO with  

`javac -cp h2o-genmodel-3.30.0.6.jar h2o_pojo_sar/XGBoost_model_python_1589382591366_94.java`

javac -cp h2o-genmodel-3.30.0.6.jar ./binary/XGBoost_grid__1_AutoML_20200717_163214_model_159.java

javac -cp h2o-genmodel-3.30.0.6.jar ./regression/drf_887c2e5b_0941_40b7_ae26_cae274c4b424.java


`drum score --code-dir ./h2o_pojo/regression --input ../../tests/testdata/boston_housing.csv`

`drum score --code-dir ./h2o_pojo/binary --input ../../tests/testdata/iris_binary_training.csv --positive-class-label 1 --negative-class-label 0`

### Score 
drum score --code-dir ./h2o_pojo_sar --input ../../tests/testdata/DR_Demo_AML_Alert.csv

### Validation 
drum validation --code-dir ./h2o_pojo_sar --input ../../tests/testdata/DR_Demo_AML_Alert.csv


### Performance Test
drum perf-test --code-dir ./h2o_pojo_sar --input ../../tests/testdata/DR_Demo_AML_Alert.csv


