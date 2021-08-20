# Model Metadata Validation Schema
The `typeSchema` is an optional part of the model_metadata.yaml file that consists of `input_requirements` and 
`output_requirements` fields.  Both requirements fields are optional, allowing the user to specify input and/or output
requirements, as appropriate to specify exactly what kinds of data the task expects or outputs.

## Purpose
The schema validation system, which is defined under the `typeSchema` field in model_metadata.yaml
is used to define the expected input and output data requirements for a given custom task.  This allows for easy communication of the expected input data from the creator to end users of a task.
The specified conditions are also used at fit time to verify that the data provided meets the 
constraints specified.  Inside DataRobot the requirements are used to verify that the blueprint inside the
project provides the required datatypes, and conforms to the sparsity, number of columns and missing values constraints
specified for the task.  

## Validation Fields
The full specification of all possible conditions and values for validation are listed in MODELING-METADATA.md.  Below is
an explanation of how to utilize the various fields available. Only a single entry for any field may be present in the
list for input requirements and for output requirements unless noted otherwise below.  

### data_types
The `data_types` field is used to specify those datatypes that are specifically expected, or those that
are specifically disallowed.  Only a single entry is supported for this field in each specification.  
The conditions used for data_types are:
- EQUALS:
  All of the listed data types are expected in the dataframe.  An error will occur if one is missing or unexpected types are found.
- IN: All of the listed data types are supported, but not all are required to be present.
- NOT_EQUALS: The datatype for the input dataframe may not be this value.
- NOT_IN: None of these datatypes are supported by the task.  

### sparse
The `sparse` field is used to define if the task supports input data that is of a sparse format, or if the
task can create output data that is in a sparse format. For input the acceptable values are:
A condition of EQUALS must always be used for sparsity specifications.

- FORBIDDEN: The task cannot handle a sparse matrix format, and will fail if one is provided.
- SUPPORTED: The task can use a sparse matrix, or a dense matrix as input.
- REQUIRED: This task only supports a sparse matrix as input and cannot use a dense dataframe.  

For the output of the task the following values are allowed:
- NEVER: This task's output is never a sparse dataframe.
- DYNAMIC: This task can output either a dense or sparse matrix.
- ALWAYS: This task will always be a sparse matrix.
- IDENTITY: This task can output either a sparse or dense matrix, and the sparsity will match the input matrix. 

### number_of_columns
The constraint for number of columns allows specifying if a specific number of columns are required or if the task has
some maximum or minimum number of columns.  For time consuming tasks specifying a maximum number of columns can help
keep performance reasonable.  Number of columns allows multiple entries to create ranges of values that are allowed.  
Note that some conditions only allow a single entry. See the [example](#example-typeSchema) below.

- EQUALS: The number of columns in the dataframe will exactly match the value.  No additional conditions allowed.
- IN:  Multiple possible acceptable values are possible.  The values are provided as a list in the value field. No additional conditions allowed.
- NOT_EQUALS: The number of columns must not be the specified value.
- GREATER_THAN: The number of columns in the dataframe must be greater than the value provided.
- LESS_THAN: The number of columns in the dataframe must be less than the value provided.
- NOT_GREATER_THAN: The number of columns in the dataframe must be less than or equal to the value provided.
- NOT_LESS_THAN: The number of columns in the dataframe must be greater than or equal to the value provided.

### contains_missing
The `contains_missing` field specifies if a task can accept missing data, or if missing data can be passed on from the
task in the output.  A condition of EQUALS will always be used. 
For the input requirements the following values are allowed:
- FORBIDDEN: The task cannot accept missing values/NA in the input dataframe. 
- SUPPORTED: The task is capable of dealing with missing values.

For output requirements the following values are allowed:
- NEVER: The task will never output missing values.
- DYNAMIC: The task might output missing values.

## Running checks locally

When running `drum fit` or `drum push` the full set of validation will automatically be run.  The first step of verification
checks that the supplied typeSchema items meet the required format.  Any format issues must be addressed before the task can be trained locally or 
on DataRobot.  After format validation the input dataset used for fit is compared against the supplied input_requirements
specifications.  Following task training the output of the task will be compared to the output_requirements and an error
reported if a mismatch is present.  

### ignoring validation
During development of a task it might be useful to disable the validation.  In that case the `--disable-strict-validation`
may be used to ignore any errors.  

## Transform Task default schema
The [default schema](custom_model_runner/datarobot_drum/resource/default_typeschema/model-metadata.yaml) is used when a 
schema isn't supplied for a task.  The default allows sparse data and missing values in the input, and the following 
datatypes:
- NUM
- CAT
- TXT
- DATE
- DATE_DURATION

The default output data type is NUM.  If any of these values are not appropriate for the task then a schema must be 
supplied in model-metadata.yaml.  


## Example typeSchema
This example would be in addition to other required fields in model_metadata.yaml.  
```yaml
- typeSchema:
  - input_requirements:
      - field:  data_types
        condition: NOT_IN
        value: 
          - TXT
          - IMG
          - DATE
      - field: sparse
        condition: EQUALS
        value:  SUPPORTED
      - field: number_of_columns
        condition: NOT_LESS_THAN
        value: 3
      - field: number_of_columns
        condition: LESS_THAN
        value: 15
      - field: contains_missing
        condition: EQUALS
        value: SUPPORTED
    - output_requirements:
        - field: data_types
          condition: EQUALS
          value: NUM
        - field: sparse
          condition: EQUALS
          value: IDENTITY
        - field: number_of_columns
          condition: EQUALS
          value: 1
        - field: contains_missing
          condition: EQUALS
          value: NEVER
```
