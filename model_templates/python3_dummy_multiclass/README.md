## Python Dummy Multiclass Inference Model Template

This classification model is a very simple example that always yields 0.75 probability for the first class, regardless of the provided input dataset.
It works with any Python environment that has `pandas`, any target & positive/negative class names can be used.
Any target and class labels work with this model.

## Instructions
Create a new custom model with this `custom.py` and use any Python Drop-In Environment with it.

### To run locally using 'drum'
Paths are relative to `./datarobot-user-models`:


`drum score --code-dir model_templates/python3_dummy_multiclass --target-type multiclass --class-labels-file model_templates/python3_dummy_multiclass/class_labels.txt --input tests/testdata/juniors_3_year_stats_regression.csv`

Note: any input dataset will work for this model.
