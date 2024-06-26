## Python Dummy Text Generation Inference Model Template

This text generation model is a very simple model that generates text output based on input.
It works with any Python environment that has `pandas`.
Expects `input` column name in the input dataset to have text. Output results are reversed text inputs.

## Instructions
Create a new custom model with this `custom.py` and use any Python Drop-In Environment with it.

### To run locally using 'drum'

Paths are relative to `./datarobot-user-models`:

`drum score --code-dir model_templates/python3_dummy_textgen --target-type textgeneration --input tests/testdata/simple_text.csv`

### To run 'drum' locally in server mode and submit request
Paths are relative to `./datarobot-user-models`:


`drum server --code-dir model_templates/python3_dummy_textgen/ --target-type textgeneration --address localhost:6789`

To submit request using `curl`:  
`curl -X POST http://localhost:6789/predictions/ -H "Content-Type: text/csv" --data-binary @/<absolute path to the file>/tests/testdata/simple_text.csv`
