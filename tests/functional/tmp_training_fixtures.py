from datarobot import CustomModelVersion, ExecutionEnvironment, ExecutionEnvironmentVersion
from datarobot.models.api_object import APIObject
from datarobot.utils import encode_utf8_if_py2
from datarobot.models.custom_model import _CustomModelBase
import trafaret as t


class CustomTrainingModel(_CustomModelBase):
    _model_type = "training"


class CustomTrainingBlueprint(APIObject):
    _path = "customLearningBlueprints/"
    _converter = t.Dict(
        {
            t.Key("blueprint_id") >> "id": t.String(),
            t.Key("custom_model"): t.Dict({t.Key("id"): t.String(), t.Key("name"): t.String()}),
            t.Key("custom_model_version"): t.Dict(
                {t.Key("id"): t.String(), t.Key("label"): t.String()}
            ),
            t.Key("execution_environment"): t.Dict(
                {t.Key("id"): t.String(), t.Key("name"): t.String()}
            ),
            t.Key("execution_environment_version"): t.Dict(
                {t.Key("id"): t.String(), t.Key("label"): t.String()}
            ),
            t.Key("training_history"): t.List(t.Dict()),
        }
    )

    def __init__(
        self,
        id,
        custom_model,
        custom_model_version,
        execution_environment,
        execution_environment_version,
        training_history,
    ):
        self.id = id
        self.custom_model = custom_model
        self.custom_model_version = custom_model_version
        self.execution_environment = execution_environment
        self.execution_environment_version = execution_environment_version
        self.training_history = training_history
        self.project_id = None

    def __repr__(self):
        return encode_utf8_if_py2(u"{}({})".format(self.__class__.__name__, self.id))

    @classmethod
    def create(
        cls,
        custom_model_id,
        environment_id,
        custom_model_version_id=None,
        environment_version_id=None,
    ):
        """Create a custom learning blueprint.
        .. versionadded:: v2.21
        Parameters
        ----------
        custom_model_id: str
            the id of the custom model
        environment_id: str
            the id of the execution environment
        custom_model_version_id: str
            the id of the custom model version
        environment_version_id: Optional[str]
            the id of the execution environment version. If version is none, what happens?
        if custom_model_id is provided, latest version will be used.
        if custom_model_version_id is provided, custom_model_id will be looked up
        One must be provided
        same for execution environments
        Returns
        -------
        CustomTrainingBlueprint
            created custom learning blueprint
        Raises
        ------
        datarobot.errors.ClientError
            if the server responded with 4xx status
        datarobot.errors.ServerError
            if the server responded with 5xx status
        """
        if not custom_model_version_id:
            model = CustomTrainingModel.get(custom_model_id=custom_model_id)
            custom_model_version_id = model.latest_version.id

        if not environment_version_id:
            ee = ExecutionEnvironment.get(execution_environment_id=environment_id)
            environment_version_id = ee.latest_version.id

        payload = {
            "custom_model_id": custom_model_id,
            "custom_model_version_id": custom_model_version_id,
            "environment_id": environment_id,
            "environment_version_id": environment_version_id,
        }
        response = cls._client.post(cls._path, data=payload)
        return cls.from_server_data(response.json())

    @classmethod
    def create_clean(cls, name, environment_dir, training_code_files, target_type):
        desc = "Generated from python client"
        # Make environment
        ee = ExecutionEnvironment.create(
            name=name + "-environment", description=desc, programming_language="python"
        )

        ev = ExecutionEnvironmentVersion.create(str(ee.id), environment_dir, description=desc)

        # Make custom model
        cm = CustomTrainingModel.create(
            name=name + "-model", target_type=target_type, description=desc
        )
        cmv = CustomModelVersion.create_clean(custom_model_id=cm.id, files=training_code_files)

        return cls.create(
            environment_id=ee.id,
            environment_version_id=ev.id,
            custom_model_id=cm.id,
            custom_model_version_id=cmv.id,
        )

    @classmethod
    def get(cls, blueprint_id):
        """Get custom learning blueprint by id.
        .. versionadded:: v2.21
        Parameters
        ----------
        blueprint_id: str
            the id of the custom learning blueprint
        Returns
        -------
        CustomTrainingBlueprint
            retrieved custom learning blueprint
        Raises
        ------
        datarobot.errors.ClientError
            if the server responded with 4xx status.
        datarobot.errors.ServerError
            if the server responded with 5xx status.
        """
        path = "{}{}/".format(cls._path, blueprint_id)
        return cls.from_location(path)
