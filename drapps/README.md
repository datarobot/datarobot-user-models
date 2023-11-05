# Custom apps hosting in DataRobot with DRApps

DRApps is a simple command line interface (CLI) providing the tools required to 
host a custom application, such as a Streamlit app, in DataRobot using a DataRobot 
execution environment. This allows you to run apps without building your own docker 
image. Custom applications don't provide any storage; however, you can access the 
full DataRobot API and other services.

## Install the DRApps CLI tool

To install the DRApps CLI tool, clone this directory (`./drapps`) from the 
[datarobot-user-models (DRUM) repository](https://github.com/datarobot/datarobot-user-models/tree/master) 
and then install the Python requirements by running the following command:

``` sh
pip install -r requirements.txt
```

## Use the DRApps CLI

After you install the DRApps CLI tool, you can use the `--help` command to 
access the following information:

``` sh
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

```

More detailed descriptions for each argument are provided in the table below:

Argument     | Description
-------------|-------------
`--base-env` | Enter the UUID or name of execution environment used as base for your Streamlit app. The execution environment contains the libraries and packages required by your application. You can find list of available environments in the **Custom Model Workshop** on the [**Environments**](https://app.datarobot.com/model-registry/custom-environments) page. <br> For a custom Streamlit application, use `--base-env '[DataRobot] Python 3.9 Streamlit'`.
`--path`     | Enter the path to a folder used to create the custom application. Files from this folder are uploaded to DataRobot and used to create the custom application image. The custom application is started from this image. <br> To use the current working directory, use `--path .`.
`--name`     | Enter the name of your custom application. This name is also used to generate the name of the custom application image, adding `Image` suffix. <br> The default value is `CustomApp`.
`--token`    | Enter your API Key, found on the [**Developer Tools**](https://app.datarobot.com/account/developer-tools) page of your DataRobot account. <br> You can also provide your API Key using the `DATAROBOT_API_TOKEN` environment variable.
`--endpoint` | Enter the URL for the DataRobot Public API. The default value is `https://app.datarobot.com/api/v2`. <br> You can also provide the URL to Public API using the `DATAROBOT_ENDPOINT` environment variable.

## Deploy an example app

To test this, deploy an example Streamlit app using the following command from 
the [`./drapps`](https://github.com/datarobot/datarobot-user-models/tree/master/drapps) directory:

``` sh
./drapps.py -t <your_api_token> -e "[Experimental] Python 3.9 Streamlit" -p ./demo-streamlit
```

This example script works as follows:

1. Finds the execution environment through the `/api/v2/executionEnvironments/` 
endpoint by the name or UUID you provided, verifying if the environment can be 
used for the custom application and retrieving the ID of the latest environment version.

2. Finds or creates the custom application image through the `/api/v2/customApplicationImages/` 
endpoint, named by adding the `Image` suffix to the provided application name (i.e., `CustomApp Image`).

3. Creates a new version of a custom application image through the `customApplicationImages/<appImageId>/versions` 
endpoint, uploading all files from the directory you provided and setting the execution 
environment version defined in the first step.

4. Starts a new application with the custom application image version created 
in the previous step.

When this script runs successfully, `Custom application successfully created` appears 
in the terminal. You can access the application on the DataRobot 
[**Applications**](https://app.datarobot.com/applications) tab.

> [!IMPORTANT]
> To access the application, you must be logged into the DataRobot instance and 
> account associated with the application.

## Considerations

Consider the following when creating a custom app:

* The root directory of the custom application must contain a `start-app.sh` file, 
used as the entry point for starting your application server.

* The web server of the application must listen on port `8080`.

* The required packages can be listed in a `requirements.txt` file in the application's 
root directory for automatic installation during application setup.

* The application can access the DataRobot API key through the `DATAROBOT_API_TOKEN`
 environment variable.

* The application can access the DataRobot Public API URL for the current environment 
through the `DATAROBOT_ENDPOINT` environment variable.