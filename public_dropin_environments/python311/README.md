# Python 3.11 Drop-In Template Environment *

This template environment can be used to create Python-based custom models.
You are responsible for installing all required dependencies. 
To do so, provide requirements.txt together with your model files

## Supported Libraries

This environment has built for Python 3 and has support for the following scientific libraries.
For specific version information, see [requirements](requirements.txt).

- datarobot-drum

## Instructions

1. From the terminal, run `tar -czvf py_dropin.tar.gz -C /path/to/public_dropin_environments/python311/ .`
2. Using either the API or from the UI create a new Custom Environment with the tarball created
in step 1.

### Creating models for this environment

Provide any files required for you model.

The structure of your custom model archive may look like:

- custom_model.tar.gz
  - artifact.pkl
  - custom.py

Please read [datarobot-cmrunner](../../custom_model_runner/README.md) documentation on how to assemble **custom.py**.
