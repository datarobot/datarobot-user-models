 # Custom GPU Environment Templates
The [public_dropin_notebook_environments](https://github.com/datarobot/datarobot-user-models/tree/master/public_dropin_notebook_environments)
contains templates for the built-in base GPU accelerated environments used in DataRobot.

The majority of these environments are focused around DataRobot's GenAI features but some can also be used in accelerating traditional deep learning models. You can extend these templates to include or modify the additional
dependencies that you want to include for your own custom environments.

In this repository, we provide several example environments that you can use and modify:
* [NVIDIA Triton Inference Server](triton_server)
* [NVIDIA NIM Example Environment](nim_llm)
* [vLLM OpenAI’s Compatible Server](vllm)
* [Python 3.11 GenAI Environment](python311_genai)

These sample environments each define the libraries available in the environment
and are designed to allow for simple custom model environments to be made that
consist solely of your custom packages if necessary.

For detailed information on how to run models that work in these environments,
reference the links above for each environment.
