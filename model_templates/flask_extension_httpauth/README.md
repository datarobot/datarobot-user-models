## Extending Web Server Behavior

This sample is meant to illustrate how you can add thirdparty or your own custom Flask extensions
when drum is in server-mode. To demonstrate one potential usecase, the `custom_flask.py` file in
this model directory will extend the HTTP server to require a specific [Bearer Token](https://swagger.io/docs/specification/authentication/bearer-authentication/)
when making any requests to it.

For completeness, we also include all the model related files from the [Python Sklearn Inference Model Template](../python3_sklearn/).

Note: it is **not** necessary (nor recommended) to add authentication to custom models that are created in DataRobot MLOps.
This example is simply to demonstration the flexibility of the `custom_flask.py` hook.

## Instructions
Create a new custom model with these files and use the Python Drop-In Environment with it.

**Important:** extending the web server is only available when running **without** the `--production` flag (or `PRODUCTION=1` environment variable).

### To run locally using 'drum'
Paths are relative to `./datarobot-user-models`:
```
drum server --docker public_dropin_environments/python3_sklearn --code-dir model_templates/python3_sklearn_flask_ext/ --target-type regression --address localhost:8080
```
