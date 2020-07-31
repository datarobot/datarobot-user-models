# Model Metadata
The `drum` tool has a verb called `push` which requires the usage of a metadata file in your 
code directory to configure the creation of a DataRobot model. 
## Options for either Training or Inference models
### Required Arguments
* name: a string used as the custom model title, try to make this unique so you can search for it 
    later.
* type: a string with the value either `training` or `inference`. Inference models are meant for
solely deployment, while training models will be able to be trained on the leaderboard. 
* environmentID: a hash of the execution environment to use while running your custom model. 
    You can find a list of available execution environments [here](https://app.datarobot.com/model-registry/custom-environments). 
    Click on the `Environment Info` tab of the environment and copy the ID to your file. 
* targetType: a string with the value either `binary` or `regression` representing the kind of 
    prediction your model is able to make
### Optional Arguments
* modelID: Once you have created a model for the first time, it is best practice to use 
custom model versions when adding code while iterating on your model. To only create a new version
instead of a whole new top level model, please include a hash here for the custom model you created.
* description: A searchable note to your future self about the contents of this model. This is 
ignored if modelID is set. 
* majorVersion: Whether the model version you are creating should be a major version update or a 
minor version update. If the previous model version is 2.3, a major version update would create the
version 3.3, and a minor version update would create the version 2.4. 

## Options specific to inference models
NOTE: All options specific to inference or training models are ignored if modelID is set- they
configure the base `custom model` entity only. However, they are still required to keep in the
metadata file.
### Required Arguments
targetName: a string with the column of your data that your model tries to predict. 
### Optional Arguments
positiveClassLabel / negativeClassLabel: Required for binary models. If your model predicts the 
 number 0, the negativeClassLabel dictates of your prediction that corresponds to. 
predictionThreshold: Optional for binary models. The cutoff point between 0 and 1 that represents
which label will be chosen as the predicted label. 

## Options specific to training models
### Optional Arguments
trainOnProject: A hash with the pid of a project you would like to train your new model or version 
on. If this is supplied, the code you supplied will start to run against this pid automagically. 
