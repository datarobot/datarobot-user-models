# Using a pretrained model in Python

This template shows how to utilize a pretrained model in a custom task.  The pretrained model
in this case is a simple linear regression model trained on the [Juniors 3 year stats dataset](tests/testdata/juniors_3_year_stats_regression.csv).  It 
is saved using `joblib.dump`, and loaded when the estimator is instantiated for prediction.

