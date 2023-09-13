## Python Dummy Regression Inference Model Template

This regression model is a very simply example that always yields fixed value 42, regardless of the provided input dataset.
It works with any Python environment that has `pandas`, any target can be set for this model.

## Instructions
Create a new custom model with this `custom.py` and use any Python Drop-In Environment with it.

### To run locally using 'drum'
Paths are relative to `./datarobot-user-models`:

`drum score --code-dir model_templates/python3_dummy_regression --target-type regression --input tests/testdata/juniors_3_year_stats_regression.csv`

Note: any input dataset will work for this model.
