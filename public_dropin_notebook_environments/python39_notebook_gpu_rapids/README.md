# Python 3.9 Notebook Drop-In Template Environment

This template environment can be used to create custom Python 3.9 notebook environments.

## Supported Libraries

This environment has been built for python 3.9 and includes commonly used OSS machine learning and data science libraries.
For specific version information, see [requirements](requirements.txt).

## Instructions

1. Update [requirements](requirements.txt) to add your custom libraries supported by Python 3.9.
2. From the terminal, run:

    ```
    tar -czvf py311_notebook_dropin.tar.gz -C /path/to/public_dropin_notebook_environments/python39_notebook_gpu_rpids/ .
    ```

3. Using either the API or from the UI create a new Custom Environment with the tarball created in step 2.

### Using this environment in notebooks

Upon successful build, the custom environment can be used in notebooks, by selecting it 
from `Session environment` > `Environment` in the notebook sidebar.
