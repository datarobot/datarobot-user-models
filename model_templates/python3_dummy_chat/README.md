## Python Dummy Text Generation Chat Model Template

This is a simple text generation model that supports OpenAI API chat() and models().

## Instructions
Create a new custom model with this `custom.py` and use any GenAI Python Drop-In Environment with it.

`drum score --code-dir model_templates/python3_chat --target-type textgeneration --input tests/testdata/simple_text.csv`

### To run 'drum' locally in server mode and submit request
Paths are relative to `./datarobot-user-models`:

`export TARGET_NAME=completion  # not used by chat, but required for starting the model`

`drum server --code-dir model_templates/python3_dummy_chat/ --target-type textgeneration --address localhost:6789`

### Using `curl`:

#### List models:
`curl localhost:6789/models`

### Using OpenAI Python client:

#### List models:

```python
from openai import OpenAI

url = "http://localhost:6789"
api_token = "not-needed"
client = OpenAI(base_url=url, api_key=api_token, _strict_response_validation=False)

models = client.models.list()
for model in models:
    print(model.to_dict())
```

#### Simple chat:

```python
from openai import OpenAI

url = "http://localhost:6789"
api_token = "not-needed"
client = OpenAI(base_url=url, api_key=api_token, _strict_response_validation=False)

response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[ {'role': 'user', 'content': 'Greetings'} ],
    temperature=0,
)
print(response)
```

