# Model Metadata Validation Schema
The documentation for the validation schema can be found [here](https://docs.datarobot.com/en/docs/modeling/special-workflows/cml/cml-ref/cml-validation.html)
in the DataRobot Docs site.  

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
