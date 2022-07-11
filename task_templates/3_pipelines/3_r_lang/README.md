## R Fit Template

The custom.R template includes a basic fit and init method that can be used to train a regression or classification model.
The expected arguments to the fit method should remain the same, although the internal functionality can be tweaked to 
use different modeling or preprocessing techniques.

Inside you will find an .init hook, which loads required libraries and
a fit hook that makes predictions. 

### To run locally using 'drum'
Paths are relative to `datarobot-user-models` root:
`drum fit --code-dir task_templates/3_pipelines/3_r_lang --input tests/testdata/juniors_3_year_stats_regression.csv --target-type regression --target "Grade 2014"`
If the command succeeds, your code is ready to be uploaded.

> Note: It may be needed to rename dataset target feature to be without a space.


