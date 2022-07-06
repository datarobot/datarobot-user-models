## Python + Multiple CodeGen models

### Build java environment
Paths are relative to the repository root - `./datarobot-user-models`:  
`cd public_dropin_environments/java_codegen`   
`docker build -t java_env .`

Then run locally

### To run locally using 'drum'
Paths are relative to the repository root - `./datarobot-user-models`:
`drum score --code-dir ./model_templates/python_multi_codejen_reg_class/ --target-type multiclass --input tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv --class-labels STAR GALAXY QSO --docker java_env --verbose`

### To run DRUM as a server
Paths are relative to the repository root - `datarobot-user-models`:  
`drum server --code-dir model_templates/python_multiple_codegen/ --target-type regression  --docker java_env --address 0.0.0.0:4567`
`drum server --code-dir ./model_templates/python_multi_codejen_reg_class/ --target-type multiclass --input tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv --class-labels STAR GALAXY QSO --docker java_env --address 0.0.0.0:4567`
To make predictions:  
`curl -X POST --form "X=@./tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv" 0.0.0.0:4567/predict/`