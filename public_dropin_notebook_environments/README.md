 # Custom Notebook Environment Templates
The [public_dropin_notebook_environments](https://github.com/datarobot/datarobot-user-models/tree/master/public_dropin_notebook_environments)
contains templates for the built-in base notebook environments used in DataRobot.

These environment templates contain the requirements needed in order for the environment to be compatible with 
both DataRobot Notebooks and Custom Models. 
You can extend these templates to include or modify the additional 
dependencies that you want to include for your own custom environments.

In this repository, we provide several example environments that you can use and modify:
* [Python 3.11 Base Notebook Environment](python311_notebook_base)

These sample environments each define the libraries available in the environment 
and are designed to allow for simple custom notebook environments to be made that 
consist solely of your custom packages if necessary.

For detailed information on how to run notebooks that work in these environments, 
reference the links above for each environment.

## CI - Pipelines in Harness to build and push to the Docker registry:

- [python311_notebook_base](python311_notebook_base) : [Publish public python notebooks environment image](https://app.harness.io/ng/account/oP3BKzKwSDe_4hCFYw_UWA/module/ci/orgs/CodeExperience/projects/NBX_Custom_Environments/pipelines/publish_public_python_notebooks_environment_image/pipeline-studio/?storeType=REMOTE&connectorRef=account.svc_harness_git1&repoName=notebooks) 
