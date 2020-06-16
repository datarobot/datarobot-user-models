# Tests
Here is how test are organized in this repo. If you want to add more test cases, decide under which category they fall, or create a new one.


## tests folder ciontent

- [fixtures](fixtures) - folder contains pre-created model artifacts, datasets, customization files used in the tests. 
- [drum](drum) - integration tests for the drum tool. Tests run in a single virtual environment which installs all the dependencies required by artifacts located in [fixtures](#fixtures). 
- [functional](functional) - contains test cases which run in custom environments:
    - [test_inference_model_templates.py](functional/test_inference_model_templates.py) - tests model_tempates in related drop-in environments. [Model templates](../model_templates) are customer facing examples which must work in the [drop-in environments](../public_dropin_environments) installed in the DataRobot App; or inside Docker containers based on Dockerfiles for those environments.
    - [test_training_model_templates.py](functional/test_training_model_templates.py) - same as inference, but for training models.
    - [test_drop_in_environments.py](functional/test_drop_in_environments.py) - arbitrary tests for models and environments. Assemble model/environment and create a test case.
        > *Note: But if you think, your generic example can be useful, create it as a model template and a appropriate environment.*
