# R Notebook Drop-In Template Environment

This template environment can be used to create custom R notebook environments.

## Supported Libraries

This environment has been built for R and supported libraries.
For specific version information, see [requirements](requirements.txt).

## Instructions

1. Update [setup-common](setup-common.R) to add your custom libraries supported by R.
2. From the terminal, run `tar -czvf r_notebook_dropin.tar.gz -C /path/to/public_dropin_notebook_environments/r_notebook/ .`
3. Using either the API or from the UI create a new Custom Environment with the tarball created in step 2.

### Using this environment in notebooks

Upon successful build, the custom environment can be used in notebooks, by selecting it 
from `ENVIRONMENT` settings > `Image` in the notebook sidebar.