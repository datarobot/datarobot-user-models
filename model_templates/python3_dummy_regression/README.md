## Python Dummy Regression Inference Model Template

This regression model is a very simple example that always yields fixed value 42 for each supplied data row, regardless of the provided input dataset.
It works with any Python environment that has `pandas`, any target can be set for this model.

## Instructions
Create a new custom model with this `custom.py` and use any Python Drop-In Environment with it.

### To run locally using 'drum'
Paths are relative to `./datarobot-user-models`:


`drum score --code-dir model_templates/python3_dummy_regression --target-type regression --input tests/testdata/juniors_3_year_stats_regression.csv`

### To run 'drum' locally in server mode and submit request
Paths are relative to `./datarobot-user-models`:


`drum server --code-dir model_templates/python3_dummy_regression/ --target-type regression --address localhost:6789`

To submit request using `curl`:  
`curl -X POST http://localhost:6789/predictions/ -H "Content-Type: text/csv" --data-binary @/<absolute path to the file>/tests/testdata/juniors_3_year_stats_regression.csv`

Note: any input dataset will work for this model.
