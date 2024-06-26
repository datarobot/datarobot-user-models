# Custom Task With Network Access And Credentials

This example shows a Binary Estimator that uses an API endpoint with credentials

## Required Feature Flags

- CUSTOM_TASKS_NETWORK_ACCESS

## Enabling Public IP Address Access

In order to have network access from within your custom task, you'll need to specifically enable it on your
Custom Task Version using the `outgoingNetworkPolicy` field.  Any new versions will inherit the previous version's 
`outgoingNetworkPolicy` unless you specify a different one. Currently, this is only settable from the public API.

### With Early-Access Python Client

`pip install datarobot_early_access`

see: https://datarobot-public-api-client.readthedocs-hosted.com/en/early-access/reference/modeling/spec/custom_task.html#create-custom-task-version

```python
from datarobot.enums import CustomTaskOutgoingNetworkPolicy

task_version = dr.CustomTaskVersion.create_clean(
    custom_task_id=custom_task_id,
    base_environment_id=execution_environment.id,
    folder_path=custom_task_folder,
    outgoing_network_policy=CustomTaskOutgoingNetworkPolicy.PUBLIC,
)
```

### Using Latest Python Client

```python

custom_task = CustomTask.get(custom_task_id)
latest_version = custom_task.latest_version
env_id = latest_version.base_environment_id
payload = {
    "baseEnvironmentId": env_id,
    "outboundNetworkPolicy": "PUBLIC"
}
client.patch(
    f"customTasks/{custom_task_id}/versions",
    json=payload,
)
custom_task.refresh()
version_id = custom_task.latest_version.id
```

## Testing In DRUM

If you want to test in DRUM with your credentials, you can fake the data by making a secrets directory and
putting all of your "secrets" there. 

See: `task_templates/2_estimators/13_python_credentials_and_internet_access/secrets` for an example

### Example Command

```shell
drum fit -cd task_templates/2_estimators/13_python_credentials_and_internet_access/ \
--input tests/testdata/10k_diabetes.csv --target-type binary --target readmitted \
--user-secrets-mount-path task_templates/2_estimators/13_python_credentials_and_internet_access/ \
--verbose --logging-level info --show-stacktrace
```

### Secrets Details
Each secret file should have the same name as one of the credentials
and the contents should be a json string that can get cast to one of the secrets objects.  All secrets objects
are in `custom_model_runner/datarobot_drum/custom_task_interfaces/user_secrets.py`.  Your secret response must 
contain a `credential_type` which is a name of `datarobot_drum.custom_task_interfaces.user_secrets.SecretType` but
in all lower case (`SecretType.SNOWFLAKE_KEY_PAIR_USER_ACCOUNT` corresponds to 
`{"credential_type": "snowflake_key_pair_user_account"}`)

```python
@dataclass(frozen=True)
class SnowflakeKeyPairUserAccountSecret(AbstractSecret):
    username: Optional[str]
    private_key_str: Optional[str]
    passphrase: Optional[str] = None
    config_id: Optional[str] = None
```

would be

```json
{
  "credential_type": "snowflake_key_pair_user_account",
  "username": "bob@bob.com",
  "private_key_str": "shhhhhhhh"
}
```