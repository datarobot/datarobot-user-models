# Model Metadata
The `drum` tool has a verb called `push` which requires the usage of a metadata file in your 
code directory to configure the creation of a DataRobot model. 
## Options for either tasks or inference models
* name (required): a string used as the custom model title, try to make this unique so you can search for it 
    later.
* type (required): a string with the value either `training` or `inference`. Inference models are meant for
solely deployment, while tasks will be able to be trained on the leaderboard. 
* environmentID (required): a hash of the execution environment to use while running your custom model. 
    You can find a list of available execution environments [here](https://app.datarobot.com/model-registry/custom-environments). 
    Click on the `Environment Info` tab of the environment and copy the ID to your file. 
* targetType (required): a string indicating the type of target.  Must be one of 
    - `binary`
    - `regression`
    - `anomaly`
    - `unstructured`
    - `multiclass`
    - `transform`
    - `textgeneration`
    - `geopoint`
    - `vectordatabase`
* modelID (optional): Once you have created a model for the first time, it is best practice to use 
custom model versions when adding code while iterating on your model. To only create a new version
instead of a whole new top level model, please include a hash here for the custom model you created.
* description (optional): A searchable note to your future self about the contents of this model. This is 
ignored if modelID is set. 
* majorVersion (optional, default: True): Whether the model version you are creating should be a 
major version update or a minor version update. If the previous model version is 2.3, a major version 
update would create the version 3.3, and a minor version update would create the version 2.4. 

## Options specific to inference models
NOTE: All options specific to inference models or tasks are ignored if modelID is set- they
configure the base `custom model` entity only. However, they are still required to keep in the
metadata file.
* targetName (required): a string with the column of your data that your model tries to predict. 
* positiveClassLabel / negativeClassLabel: Required for binary models. If your model predicts the 
 number 0, the negativeClassLabel dictates of your prediction that corresponds to. 
* predictionThreshold: Optional for binary models. The cutoff point between 0 and 1 that represents
which label will be chosen as the predicted label. 

## Options specific to tasks
* trainOnProject (optional): A hash with the pid of a project you would like to train your new model or version 
on. If this is supplied, the code you supplied will start to run against this pid automagically. 
* userCredentialSpecifications (optional): This is a list of credentials that will be injected from 
DataRobot on both fit and predict. You can get your credential IDs by looking at the URL when you click on
a credential in datarobot.com/account/credentials-management. They have the following template

```
userCredentialSpecifications:
  - key: REQUIRED - a POSIX compatable environment name (^[_a-zA-Z][_a-zA-Z0-9]*$)
    valueFrom: REQUIRED - a valid object id pointing to your credential
    reminder: OPTIONAL - any string to help you remember what this is.
```


### Validation Schema
The documentation for the validation schema can be found [here](https://docs.datarobot.com/en/docs/modeling/special-workflows/cml/cml-ref/cml-validation.html)
in the DataRobot Docs site.  
