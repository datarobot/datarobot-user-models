## Python + Multiple CodeGen models

### Build java environment
Paths are relative to the repository root - `./datarobot-user-models`:  
`cd public_dropin_environments/java_codegen`   
`docker build -t java_env .`

Then run locally

### To run locally using 'drum'
Paths are relative to the repository root - `./datarobot-user-models`:  
`drum score --code-dir model_templates/python_multiple_codegen/ --target-type regression --input tests/testdata/juniors_3_year_stats_regression.csv --docker java_env`

### To run DRUM as a server
Paths are relative to the repository root - `datarobot-user-models`:  
`drum server --code-dir model_templates/python_multiple_codegen/ --target-type regression  --docker java_env --address 0.0.0.0:4567`  
To make predictions:  
`curl -X POST --form "X=@./tests/testdata/juniors_3_year_stats_regression.csv" 0.0.0.0:4567/predict/`