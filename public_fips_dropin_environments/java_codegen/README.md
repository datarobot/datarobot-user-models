# Java Drop-In Template Environment

This template can be used as an environment for DataRobot generated scoring code or models that implement the either the `IClassificationPredictor`
or `IRegressionPredictor` interface from the [datarobot-prediction](https://mvnrepository.com/artifact/com.datarobot/datarobot-prediction) package.

## Requirements

- Java 11 JDK
- A valid scoring code jar

## Instructions

1. From the terminal, run `tar -czvf java_dropin.tar.gz -C /path/to/public_dropin_enironments/java_codegen/ .`
2. Using either the API or from the UI create a new Custom Environment with the tarball created
in step 1.

### Creating models for this environment

The model for this environment is a scoring code jar.
Using either the API or from the UI create a new Custom Model with the jar file.
