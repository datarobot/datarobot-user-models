# Custom Notebook Environment Templates

The [public_dropin_notebook_environments](https://github.com/datarobot/datarobot-user-models/tree/master/public_dropin_notebook_environments)
contains templates for the built-in base notebook environments used in DataRobot.

These environment templates contain the requirements needed in order for the environment to be compatible with 
both DataRobot Notebooks and Custom Models. 
You can extend these templates to include or modify the additional 
dependencies that you want to include for your own custom environments.

In this repository, we provide several example environments that you can use and modify:
* [Python 3.11 Base Notebook Environment](python311_notebook_base)
* [Python 3.9 Notebook Environment for GPU](python39_notebook_gpu)
* [Python 3.9 Notebook Environment for GPU with Tensorflow](python39_notebook_gpu_tf)
* [Python 3.9 Notebook Environment for GPU with Rapids](python39_notebook_gpu_rapids)
* [R Notebook Environment](r_notebook)

These sample environments each define the libraries available in the environment 
and are designed to allow for simple custom notebook environments to be made that 
consist solely of your custom packages if necessary.

For detailed information on how to run notebooks that work in these environments, 
reference the links above for each environment.

### Example base tarball

There is a pre-made tarball called `py311_notebook_dropin.tar.gz` in this `public_dropin_notebook_environments` directory. This can be used to create a simple custom environment.

This was generated using the following command:

    ```
    tar -czvf py311_notebook_dropin.tar.gz -C /path/to/public_dropin_notebook_environments/python311_notebook_base/ .
    ```

### Using environments in notebooks

Please see [DataRobot documentation](https://docs.datarobot.com/en/docs/workbench/wb-notebook/wb-code-nb/wb-env-nb.html#custom-environment-images) for more information.
