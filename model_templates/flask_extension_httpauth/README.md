## Extending Web Server Behavior

This example illustrates how you can add custom Flask extensions when DRUM is in server mode. To
demonstrate one potential use case, the `custom_flask.py` file in this model directory extends
the HTTP server to require a specific [Bearer Token](https://swagger.io/docs/specification/authentication/bearer-authentication/)
when making any requests to it. For completeness, this directory also includes all model-related
files from the [Python Sklearn Inference Model Template](../python3_sklearn/).

**Note**: It is _not_ necessary (or recommended) to add authentication to custom models that will
be deployed in DataRobot MLOps — the platform will layer its standard API authentication on top of
your custom model for you. This example is simply to demonstrate the flexibility of the
`custom_flask.py` hook or to show how you can add authentication to a 'drum' model that is served
externally.

**Important**: You can only extend the web server when running it _without_ the `--production` flag
(or `PRODUCTION=1` environment variable).

To run this example locally, use the following command with paths relative to `./datarobot-user-models`:

```
drum server --docker public_dropin_environments/python3_sklearn --code-dir model_templates/python3_sklearn_flask_ext/ --target-type regression --address localhost:8080
```
