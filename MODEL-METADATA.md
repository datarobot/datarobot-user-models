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

### Validation Schema
The validation schema is used to define input and output requirements for the task (See detailed documentation in VALIDATION-SCHEMA.md).  The validation is used to
communicate the acceptable inputs for the model along with the expected output.  This will be verified when running `drum fit`
Detailed information about using schema validation can be found [here](#VALIDATION-SCHEMA.md), along with examples the task templates.
* typeSchema (optional): Top level dictionary that contains the input and output schema definitions
  * input_requirements (optional):  Specifications that apply to the models input.  The specifications provided as a list.
  * output_requirements (optional): Specifications that define the expected output of the model. The specifications provided as a list.
    
All specifications contain the following fields:
* field: which specification is being defined, one of `data_types`, `sparse`, `number_of_columns`, `contains_missing`
* condition: defines how the values in the `value` field are used
* value: A list or single value, depending upon the condition used

#### data_types allowed values:
- condition: "EQUALS", "IN", "NOT_EQUALS", "NOT_IN"
- value: "NUM", "TXT", "CAT", "IMG", "DATE", "DATE_DURATION", "COUNT_DICT", "GEO",

#### sparse (input) allowed values:
- condition: "EQUALS"
- value: "FORBIDDEN", "SUPPORTED", "REQUIRED"

#### sparse (output) allowed values:
- condition: "EQUALS"
- value: "NEVER", "DYNAMIC", "ALWAYS", "IDENTITY"

#### number_of_columns allowed values:
- condition: "EQUALS", "IN", "NOT_EQUALS", "NOT_IN", "GREATER_THAN", "LESS_THAN", "NOT_GREATER_THAN", "NOT_LESS_THAN"
- value: Integer value >= 0 (for value of 0 only GREATER_THAN is allowed)
   
#### contains missing (input) allowed values
- condition: "EQUALS"
- value: "FORBIDDEN", "SUPPORTED"

#### contains missing (output) allowed values
- condition: "EQUALS"
- value: "NEVER", "DYNAMIC"