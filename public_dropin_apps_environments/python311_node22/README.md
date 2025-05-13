# Python 3.11 with NodeJS 22.15 Base Notebook Drop-In Template Environment

This template environment can be used to create custom notebook environments based on Python 3.11 and Node.js 22.15.

## Supported Libraries

This environment is built for Python 3.11 and includes only the minimal required dependencies. It also supports managing React-based applications with NodeJS for use within DataRobot Notebooks.

## Instructions

1. Update [requirements](requirements.txt) to add your custom libraries supported by Python 3.11.
2. From the terminal, run:

   ```
   tar -czvf py311_notebook_dropin.tar.gz -C /path/to/public_dropin_notebook_environments/python311_notebook_base/ .
   ```

3. Using either the API or from the UI create a new Custom Environment with the tarball created in step 2.

### Using this environment in notebooks

Upon successful build, the custom environment can be used in notebooks, by selecting it
from `Session environment` > `Environment` in the notebook sidebar.

Please see [DataRobot documentation](https://docs.datarobot.com/en/docs/workbench/wb-notebook/wb-code-nb/wb-env-nb.html#custom-environment-images) for more information.
