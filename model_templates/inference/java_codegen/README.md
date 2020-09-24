## Java Model Template

This model is intended to work with the [Java Drop-In Environment](../../../public_dropin_environments/java_codegen/).
The supplied jar is a DataRobot scoring code model trained from [boston_housing.csv](../../../tests/testdata/boston_housing.csv)
with a MEDV as the target (regression), though any jar containing a model that implements the `IClassificationPredictor` or
`IRegressionPredictor` interface from the [datarobot-prediction](https://mvnrepository.com/artifact/com.datarobot/datarobot-prediction)
package will work.

## Instructions
Upload the jar file as the only file in the custom model and use the Java Drop-In Environment with it

### To run locally using 'drum'
Paths are relative to `./datarobot-user-models`:  
`drum score --code-dir model_templates/inference/java_codegen --target-type regression --input tests/testdata/boston_housing.csv`