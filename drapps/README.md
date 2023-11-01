# DRApps

## About 

DRApps is simple CLI tool for creating your custom application in DataRobot.

Custom applications does not provide any storage but you may access the full DataRobot API and other Internet services

## Installation

For using this CLI script you need only install python requirements with:

    pip install -r requirements.txt

## Usage

    $ ./drapps.py --help
    Usage: drapps.py [OPTIONS]
    
      App that uses local file for create new custom application
    
    Options:
      -e, --base-env TEXT   Name or ID for execution environment  [required]
      -p, --path DIRECTORY  Path to folder with files that should be uploaded
                            [required]
      -n, --name TEXT       Name for new custom application. Default: CustomApp
      -t, --token TEXT      Pubic API access token.
      -E, --endpoint TEXT   Data Robot Public API endpoint
      --help                Show this message and exit.

### --base-env

UUID or name of Execution Environment that will be used as base for your app setup.
Execution Environment contains preinstalled essential libraries and packages, that 
can be used by your application.

You can find list of available Execution Environments in **Custom Model Workshop**
on [**Environments page**](https://app.datarobot.com/model-registry/custom-environments)

### --path

Path to folder which will be used to create custom application. Files from this folder
will be uploaded to Data Robot and used for creating custom application image. Custom
application will be started from this image.

### --name

Name for your custom application. Default value is `CustomApp`. This name will be also
used for generating name for custom application image by adding `Image` suffix.

### --token

API Key. May be found on [Developer Tools](https://app.datarobot.com/account/developer-tools)
tab on your  **account** page.

Also, you can provide token for the script through `DATAROBOT_API_TOKEN` environment
variable.

### --endpoint

URL to Public API of Data Robot. Default value: `https://app.datarobot.com/api/v2`.

You can provide URL to Public API through  `DATAROBOT_ENDPOINT` environment variable.

## Script work description

Script works in 4 steps:

1. Script uses `/api/v2/executionEnvironments/` endpoints for finding Execution
   Environment by name or UUID provided by user; checks if environment can be used
   for custom application and takes ID of latest version.

2. Script uses `/api/v2/customApplicationImages/` endpoint for finding or creating
   custom application image with name formed by adding `Image` suffix to application
   name.

3. Script uses `customApplicationImages/<appImageId>/versions` endpoints for creating
   new custom application image version, uploading to this version all files from
   folder provided by user and setting Execution Environment version found earlier.

4. Newly created custom application image version is used to starting a new application.

Result of script work you can find on [Applications](https://app.datarobot.com/applications)
tab.

## Tips for your custom application

Root folder of custom application must contain `start-app.sh` file, that will be
used as entry point for starting your application server.

Web server of the application must listen on **8080** port.

Required packages may be listed in `requirements.txt` in root directory. Packages
from this file will be installed automatically during application setup.

Application can get access to Data Robot API token through environment variable
`DATAROBOT_API_TOKEN`.

Application can get URL to Data Robot Public API for current environment through
environment variable `DATAROBOT_ENDPOINT`.
