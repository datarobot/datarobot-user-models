import os
import re
from pathlib import Path

import datarobot as dr_client

from datarobot_drum.drum.common import RunMode, MODEL_CONFIG_FILENAME, TargetType
from datarobot_drum.drum.exceptions import DrumCommonException

DR_LINK_FORMAT = "{}/model-registry/custom-models/{}"
MODEL_LOGS_LINK_FORMAT = "{url}/projects/{project_id}/models/{model_id}/log"


def _get_metadata(options):
    code_dir = Path(options.code_dir)
    if options.model_config is None:
        raise DrumCommonException(
            "You must have a file with the name {} in the directory {}. \n"
            "You don't. \nWhat you do have is these files: \n{} ".format(
                MODEL_CONFIG_FILENAME, code_dir, os.listdir(code_dir)
            )
        )
    return options.model_config


def _convert_target_type(unconverted_target_type):
    if unconverted_target_type == TargetType.REGRESSION.value:
        return dr_client.TARGET_TYPE.REGRESSION
    elif unconverted_target_type == TargetType.BINARY.value:
        return dr_client.TARGET_TYPE.BINARY
    elif unconverted_target_type == TargetType.ANOMALY.value:
        return dr_client.enums.CUSTOM_MODEL_TARGET_TYPE.ANOMALY
    elif unconverted_target_type == TargetType.MULTICLASS.value:
        return dr_client.enums.TARGET_TYPE.MULTICLASS
    raise DrumCommonException("Unsupported target type {}".format(unconverted_target_type))


def _push_training(model_config, code_dir, endpoint=None, token=None):
    try:
        from datarobot._experimental import CustomTrainingBlueprint, CustomTrainingModel
    except ImportError:
        raise DrumCommonException(
            "You tried to run custom training models using a version of the \n"
            "datarobot client which doesn't have this beta functionality yet. \n"
            "Please pip install datarobot>=2.22.0b0 to access this functionality. \n"
            "This requires adding the internal datarobot artifactory index \n"
            "as your pip index. "
        )
    dr_client.Client(token=token, endpoint=endpoint)
    if "modelID" in model_config:
        model_id = model_config["modelID"]
    else:
        model_id = CustomTrainingModel.create(
            name=model_config["name"],
            target_type=_convert_target_type(model_config["targetType"]),
            description=model_config.get("description", "Pushed from DRUM"),
        ).id
        print(
            "You just created a new custom model. Please add this model ID to your metadata file "
            "by adding the line 'modelID:{}'".format(model_id)
        )

    try:
        model_version = dr_client.CustomModelVersion.create_clean(
            model_id,
            base_environment_id=model_config["environmentID"],
            folder_path=code_dir,
            is_major_update=model_config.get("majorVersion", True),
        )
    except dr_client.errors.ClientError as e:
        print("Error adding model with ID {} and dir {}: {}".format(model_id, code_dir, str(e)))
        raise SystemExit(1)

    # TODO: Update this once the datarobot client is updated
    payload = dict(custom_model_version_id=model_version.id)
    response = dr_client.client.get_client().post("customTrainingBlueprints/", data=payload)
    user_blueprint_id = response.json()["userBlueprintId"]

    print("A user blueprint was created with the ID {}".format(user_blueprint_id))

    _print_model_started_dialogue(model_id)

    if "trainOnProject" in model_config.get("trainingModel", ""):
        try:
            pid = model_config["trainingModel"]["trainOnProject"]
            current_task = "fetching the specified project {}".format(pid)
            project = dr_client.Project(pid)

            # TODO: Update this once the datarobot client is updated
            payload = dict(project_id=pid, user_blueprint_ids=[user_blueprint_id])
            current_task = "adding your model to the menu"
            response = dr_client.client.get_client().post("userBlueprints/addToMenu/", data=payload)
            blueprint_id = response.json()[user_blueprint_id]

            current_task = "actually training of blueprint {}".format(blueprint_id)
            model_job_id = project.train(blueprint_id)
            lid = dr_client.ModelJob.get(project_id=pid, model_job_id=model_job_id).model_id
        except dr_client.errors.ClientError as e:
            print("There was an error training your model while {}: {}".format(current_task, e))
            raise SystemExit(1)
        print("\nIn addition...")
        print("Model training has started! Follow along at this link: ")
        print(
            MODEL_LOGS_LINK_FORMAT.format(
                url=re.sub(r"/api/v2/?", "", dr_client.client._global_client.endpoint),
                model_id=lid,
                project_id=model_config["trainingModel"]["trainOnProject"],
            )
        )


def _push_inference(model_config, code_dir, token=None, endpoint=None):
    dr_client.Client(token=token, endpoint=endpoint)
    if "inferenceModel" not in model_config:
        raise DrumCommonException(
            "You must include the inferenceModel top level key for custom infernece models"
        )
    if "targetName" not in model_config["inferenceModel"]:
        raise DrumCommonException(
            "For inference models, you must include targetName under the inferenceModel key"
        )
    if "modelID" in model_config:
        model_id = model_config["modelID"]
    else:
        create_params = dict(
            name=model_config["name"],
            target_type=_convert_target_type(model_config["targetType"]),
            target_name=model_config["inferenceModel"]["targetName"],
            description=model_config.get("description", "Pushed from DRUM"),
        )
        if model_config["targetType"] == TargetType.BINARY.value:
            create_params.update(
                dict(
                    positive_class_label=model_config["inferenceModel"].get("positiveClassLabel"),
                    negative_class_label=model_config["inferenceModel"].get("negativeClassLabel"),
                    prediction_threshold=model_config["inferenceModel"].get("predictionThreshold"),
                )
            )
        elif model_config["targetType"] == TargetType.MULTICLASS.value:
            class_labels = model_config["inferenceModel"].get("classLabels")
            class_labels_file = model_config["inferenceModel"].get("classLabelsFile")
            if not ((class_labels is None) ^ (class_labels_file is None)):
                raise DrumCommonException(
                    "Multiclass inference models must specify either classLabels or classLabelsFile"
                )
            if class_labels_file:
                with open(class_labels_file) as f:
                    class_labels = f.read().split(os.linesep)
            create_params.update(dict(class_labels=class_labels))
        model_id = dr_client.CustomInferenceModel.create(**create_params).id
    dr_client.CustomModelVersion.create_clean(
        custom_model_id=model_id,
        base_environment_id=model_config["environmentID"],
        folder_path=code_dir,
        is_major_update=model_config.get("majorVersion", True),
    )
    _print_model_started_dialogue(model_id)


def _print_model_started_dialogue(new_model_id):
    print("\nYour model was successfully pushed")
    print("\n🏁 Follow this link to get started 🏁")
    print(
        DR_LINK_FORMAT.format(
            re.sub(r"/api/v2/?", "", dr_client.client._global_client.endpoint), new_model_id
        )
    )


def _setup_training_validation(config, options):
    # Setting default values for most of these :)
    path = Path(config["validation"]["input"])
    if not os.path.isabs(path):
        path = Path(options.code_dir).joinpath(path)

    options.input = path
    options.output = None
    options.negative_class_label = None
    options.positive_class_label = None
    options.target_csv = None
    options.target = config["validation"].get("targetName")
    options.row_weights = None
    options.row_weights_csv = None
    options.num_rows = "ALL"
    options.skip_predict = False

    raw_args_for_docker = (
        "drum {run_mode} --input {input} --target {target} --code-dir {code_dir}".format(
            run_mode=RunMode.FIT, input=path, target=options.target, code_dir=options.code_dir
        ).split()
    )

    return options, RunMode.FIT, raw_args_for_docker


def _setup_inference_validation(config, options):
    path = Path(config["validation"]["input"])
    if not os.path.isabs(path):
        path = Path(options.code_dir).joinpath(path)

    options.input = path
    options.output = "/dev/null"
    options.negative_class_label = None
    options.positive_class_label = None
    raw_args_for_docker = "drum {run_mode} --input {input} -cd {code_dir}".format(
        run_mode=RunMode.SCORE, input=path, code_dir=options.code_dir
    ).split()
    return options, RunMode.SCORE, raw_args_for_docker


def setup_validation_options(options):
    model_config = _get_metadata(options)
    if model_config["type"] == "training":
        return _setup_training_validation(model_config, options)
    elif model_config["type"] == "inference":
        return _setup_inference_validation(model_config, options)
    else:
        raise DrumCommonException("Unsupported type")


def drum_push(options):
    model_config = _get_metadata(options)

    if model_config["type"] == "training":
        _push_training(model_config, options.code_dir)

    elif model_config["type"] == "inference":
        _push_inference(model_config, options.code_dir)
    else:
        raise DrumCommonException("Unsupported type")


PUSH_HELP_TEXT = """
This submits the contents of a directory as a custom model to DataRobot.

To use this functionality, you must create two types of configuration.

1. **DataRobot client configuration**
    `push` relies on correct global configuration of the client to access 
    a DataRobot server. There are two options for supplying this configuration, 
    through environment variables or through a config file which is read 
    by the DataRobot client. Both of these options will include an endpoint
    and an API token to authenticate the requests.

    * Option 1: Environment variables.

    Example:
    ```
    export DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2
    export DATAROBOT_API_TOKEN=<yourtoken>
    ```
    * Option 2: Create this file, which we check for: 
        `~/.config/datarobot/drconfig.yaml`
        
    Example:
    ```
    endpoint: https://app.datarobot.com/api/v2
    token: <yourtoken>
    ```
2. **Model Metadata** `push` also relies on a metadata file, which is 
    parsed on drum to create the correct sort of model in DataRobot. 
    This metadata file includes quite a few options. You can
    [read about those options](MODEL-METADATA.md) or 
    [see an example](model_templates/inference/python3_sklearn/model-metadata.yaml)
"""
