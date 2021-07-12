## Flux Model 

Source the appropriate Julia Environment

`export JULIA_PROJECT=/Users/timothy.whittaker/Desktop/mlops-experiments/python-julia-sys-image`

`julia --project=$JULIA_PROJECT`

Install `Flux` and `BSON` in your Julia environment if not already available

Again, this is a slow startup time.  For scoring and serving it is fine, but it will always take some time to start everything up, therefore the recommendation is to use a system image of your environment to speed up things considerably.  

## Scoring

drum score --code-dir model_templates/inference/julia/jl_custom --target-type regression --input tests/testdata/boston_housing_inference.csv --verbose --logging-level info

## Serving with Docker

drum server --code-dir model_templates/inference/julia/jl_custom --target-type regression --input tests/testdata/boston_housing_inference.csv --verbose --logging-level info






