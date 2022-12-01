 # Custom Notebook Environment Templates
The [public_dropin_notebook_environments](https://github.com/datarobot/datarobot-user-models/tree/master/public_dropin_notebook_environments)
contains templates for the base notebook environments used in DataRobot.
Dependency requirements can be applied to the base environment to create a
runtime environment for both notebooks and/or custom inference models.
A custom environment defines the runtime environment for either a custom task 
or custom inference model.
In this repository, we provide several example environments that you can use and modify:
* [Python 3.9 Notebook](python39_notebook)
* [Python 3.8 + Snowflake Notebook](python38_notebook_snowflake)
* [R Notebook](r_notebook)

These sample environments each define the libraries available in the environment 
and are designed to allow for simple custom notebook environments to be made that 
consist solely of your custom packages if necessary.

For detailed information on how to run notebooks that work in these environments, 
reference the links above for each environment.