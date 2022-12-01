# Python 3.9 Notebook Drop-In Template Environment

This template environment can be used to create custom Python 3.9 notebook environments.

## Supported Libraries

This environment has been built for python 3.9 and supported libraries.
For specific version information, see [requirements](requirements.txt).

## Instructions

1. Update [requirements](requirements.txt) to add your custom libraries supported by Python 3.9.
2. From the terminal, run `tar -czvf py39_notebook_dropin.tar.gz -C /path/to/public_dropin_notebook_environments/python39_notebook/ .`
3. Using either the API or from the UI create a new Custom Environment with the tarball created in step 2.

### Using this environment in notebooks

Upon successful build, the custom environment can be used in notebooks, by selecting it from `ENVIRONMENT` settings > `Image`.