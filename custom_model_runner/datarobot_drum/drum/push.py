import os
import re
from datarobot_drum.drum.exceptions import DrumCommonException
from pathlib import Path

import datarobot as dr_client
from strictyaml import Bool, Int, load, Map, Optional, Str, YAMLError

from datarobot_drum.drum.exceptions import DrumCommonException

CONFIG_FILENAME = "model-metadata.yaml"
DR_LINK_FORMAT = "{}/model-registry/custom-models/{}"
MODEL_LOGS_LINK_FORMAT = "{url}/projects/{project_id}/models/{model_id}/log"

schema = Map(
    {
        "name": Str(),
        "type": Str(),
        "environmentID": Str(),
        "targetType": Str(),
        "validation": Map({"inputData": Str(), "targetName": Str()}),
        Optional("modelID"): Str(),
        Optional("description"): Str(),
        Optional("majorVersion"): Bool(),
        Optional("inferenceModel"): Map(
            {
                "targetName": Str(),
                Optional("positiveClassLabel"): Str(),
                Optional("negativeClassLabel"): Str(),
                Optional("predictionThreshold"): Int(),
            }
        ),
        Optional("trainingModel"): Map({Optional("trainOnProject"): Str()}),
    }
)


def _read_metadata(code_dir):
    code_dir = Path(code_dir)
    if not code_dir.joinpath(CONFIG_FILENAME).exists():
        raise DrumCommonException(
            "You must have a file with the name {} in the directory {}. \n"
            "You don't. \nWhat you do have is these files: \n{} ".format(
                CONFIG_FILENAME, code_dir, os.listdir(code_dir)
            )
        )
    with open(code_dir.joinpath(CONFIG_FILENAME)) as f:
        try:
            model_config = load(f.read(), schema).data
        except YAMLError as e:
            print(e)
            raise SystemExit()
    return model_config


def _convert_target_type(unconverted_target_type):
    if unconverted_target_type == "regression":
        return dr_client.TARGET_TYPE.REGRESSION
    elif unconverted_target_type == "binary":
        return dr_client.TARGET_TYPE.BINARY
    raise DrumCommonException("Unsupported target type {}".format(unconverted_target_type))


def push_training(model_config, code_dir, endpoint=None, token=None):
    try:
        from datarobot._experimental import CustomTrainingBlueprint, CustomTrainingModel
    except ImportError:
        raise DrumCommonException(
            "You tried to run custom training models using a version of the "
            "datarobot client which doesn't have this beta functionality yet. "
            "Please pip install datarobot>=2.22.0b0 to access this functionality. "
            "This requires adding the internal datarobot artifactory index as your pip index. "
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

    try:
        dr_client.CustomModelVersion.create_clean(model_id, folder_path=code_dir)
    except dr_client.errors.ClientError as e:
        print("Error adding model with ID {} and dir {}: {}".format(model_id, code_dir, str(e)))
        raise SystemExit(1)

    blueprint = CustomTrainingBlueprint.create(
        environment_id=model_config["environmentID"], custom_model_id=model_id,
    )

    print("A blueprint was created with the ID {}".format(blueprint.id))

    print_model_started_dialogue(model_id)

    if "trainOnProject" in model_config.get("trainingModel", ""):
        try:
            project = dr_client.Project(model_config["trainingModel"]["trainOnProject"])
            project.train(blueprint)
        except dr_client.errors.ClientError as e:
            print("There was an error training your model: {}".format(e))
            raise SystemExit()
        print("\nIn addition...")
        print("Model training has started! Follow along at this link: ")
        print(
            MODEL_LOGS_LINK_FORMAT.format(
                url=re.sub(r"/api/v2/?", "", dr_client.client._global_client.endpoint),
                model_id=model_id,
                project_id=model_config["trainingModel"]["trainOnProject"],
            )
        )


def push_inference(model_config, code_dir, token=None, endpoint=None):
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
        model_id = dr_client.CustomInferenceModel.create(
            name=model_config["name"],
            target_type=_convert_target_type(model_config["targetType"]),
            target_name=model_config["inferenceModel"]["targetName"],
            description=model_config.get("description", "Pushed from DRUM"),
            positive_class_label=model_config["inferenceModel"].get("positiveClassLabel"),
            negative_class_label=model_config["inferenceModel"].get("negativeClassLabel"),
            prediction_threshold=model_config["inferenceModel"].get("predictionThreshold"),
        ).id
    dr_client.CustomModelVersion.create_clean(custom_model_id=model_id, folder_path=code_dir)
    print_model_started_dialogue(model_id)


def print_model_started_dialogue(new_model_id):
    print("\nYour model was successfully pushed")
    print("\n🏁 Follow this link to get started 🏁")
    print(
        DR_LINK_FORMAT.format(
            re.sub(r"/api/v2/?", "", dr_client.client._global_client.endpoint), new_model_id
        )
    )


def drum_push(options):
    model_config = _read_metadata(options.code_dir)

    if model_config["type"] == "training":
        push_training(model_config, options.code_dir)

    elif model_config["type"] == "inference":
        push_inference(model_config, options.code_dir)
    else:
        raise DrumCommonException("Unsupported type")
