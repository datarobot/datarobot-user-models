## Python Dummy Binary Classification Inference Model Template

This binary classification model is a very simple example that always yields 0.75 probability for the positive class and 0.25 for the negative class, regardless of the provided input dataset.
It works with any Python environment that has `pandas`, any target & positive/negative class names can be used.

## Instructions
Create a new custom model with this `custom.py` and use any Python Drop-In Environment with it.

### To run locally using 'drum'
Paths are relative to `./datarobot-user-models`:


`drum score --code-dir model_templates/python3_dummy_binary --target-type binary --positive-class-label true --negative-class-label false --input tests/testdata/juniors_3_year_stats_regression.csv`

Note: any input dataset will work for this model.
