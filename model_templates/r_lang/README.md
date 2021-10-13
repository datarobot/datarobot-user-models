## R Inference Model Template

This model is intended to work with the [R Drop-In Environment](../../public_dropin_environments/r_lang/).
The supplied rds file is a caret trained BRNN model trained on [juniors_3_year_stats_regression.csv](../../tests/testdata/juniors_3_year_stats_regression.csv)
with a `Grade 2014` as the target (regression), though any binary or regression model trained using the libraries
outlined in [R Drop-In Environment](../../public_dropin_environments/r_lang/) will work.

For this sample model, custom.R loads the BRNN library, which is required by the model to make predictions.

## Instructions
Create a new custom model with these files and use the R Drop-In Environment with it

### To run locally using 'drum'
Paths are relative to `./datarobot-user-models`:   
`drum score --code-dir model_templates/r_lang --target-type regression --input tests/testdata/juniors_3_year_stats_regression.csv`