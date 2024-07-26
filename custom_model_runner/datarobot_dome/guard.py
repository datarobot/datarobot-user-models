#
# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# DataRobot, Inc. Confidential.
#
# This is unpublished proprietary source code of DataRobot, Inc.
# and its affiliates.
#
# The copyright notice above does not evidence any actual or intended
# publication of such source code.
import json
import os
from abc import ABC

import datarobot as dr
import trafaret as t
from dmm.custom_metric import CustomMetricAggregationType
from dmm.custom_metric import CustomMetricDirectionality
from llama_index.core import ServiceContext
from llama_index.core.evaluation import FaithfulnessEvaluator
from nemoguardrails import LLMRails
from nemoguardrails import RailsConfig

from datarobot_dome.constants import CUSTOM_METRIC_DESCRIPTION_SUFFIX
from datarobot_dome.constants import DATE_FORMAT_STRINGS
from datarobot_dome.constants import DEFAULT_GUARD_PREDICTION_TIMEOUT_IN_SEC
from datarobot_dome.constants import DEFAULT_PROMPT_COLUMN_NAME
from datarobot_dome.constants import DEFAULT_RESPONSE_COLUMN_NAME
from datarobot_dome.constants import FAITHFULLNESS_COLUMN_NAME
from datarobot_dome.constants import NEMO_GUARD_COLUMN_NAME
from datarobot_dome.constants import NEMO_GUARDRAILS_DIR
from datarobot_dome.constants import ROUGE_1_COLUMN_NAME
from datarobot_dome.constants import SECRET_DEFINITION_PREFIX
from datarobot_dome.constants import SECRET_DEFINITION_SUFFIX
from datarobot_dome.constants import TOKEN_COUNT_COLUMN_NAME
from datarobot_dome.constants import GuardAction
from datarobot_dome.constants import GuardLLMType
from datarobot_dome.constants import GuardModelTargetType
from datarobot_dome.constants import GuardOperatorType
from datarobot_dome.constants import GuardStage
from datarobot_dome.constants import GuardTimeoutAction
from datarobot_dome.constants import GuardType
from datarobot_dome.constants import OOTBType
from datarobot_dome.guard_helpers import get_azure_openai_client

MAX_GUARD_NAME_LENGTH = 255
MAX_COLUMN_NAME_LENGTH = 255
MAX_GUARD_COLUMN_NAME_LENGTH = 255
MAX_GUARD_MESSAGE_LENGTH = 4096
MAX_GUARD_DESCRIPTION_LENGTH = 4096
OBJECT_ID_LENGTH = 24
MAX_REGEX_LENGTH = 255
MAX_URL_LENGTH = 255
MAX_TOKEN_LENGTH = 255
NEMO_THRESHOLD = "TRUE"
MAX_GUARD_CUSTOM_METRIC_BASELINE_VALUE_LIST_LENGTH = 5


model_info_trafaret = t.Dict(
    {
        t.Key("class_names", to_name="class_names", optional=True): t.List(
            t.String(max_length=MAX_COLUMN_NAME_LENGTH)
        ),
        t.Key("model_id", to_name="model_id", optional=True): t.String(max_length=OBJECT_ID_LENGTH),
        t.Key("input_column_name", to_name="input_column_name", optional=False): t.String(
            max_length=MAX_COLUMN_NAME_LENGTH
        ),
        t.Key("target_name", to_name="target_name", optional=False): t.String(
            max_length=MAX_COLUMN_NAME_LENGTH
        ),
        t.Key(
            "replacement_text_column_name", to_name="replacement_text_column_name", optional=True
        ): t.Or(t.String(allow_blank=True, max_length=MAX_COLUMN_NAME_LENGTH), t.Null),
        t.Key("target_type", to_name="target_type", optional=False): t.Enum(
            *GuardModelTargetType.ALL
        ),
    },
    allow_extra=["*"],
)


model_guard_intervention_trafaret = t.Dict(
    {
        t.Key("comparand", to_name="comparand", optional=False): t.Or(
            t.String(max_length=MAX_GUARD_NAME_LENGTH),
            t.Float(),
            t.Bool(),
            t.List(t.String(max_length=MAX_GUARD_NAME_LENGTH)),
            t.List(t.Float()),
        ),
        t.Key("comparator", to_name="comparator", optional=False): t.Enum(*GuardOperatorType.ALL),
    },
    allow_extra=["*"],
)

modifier_guard_intervention_trafaret = t.Dict(
    {
        t.Key("match_regex", to_name="match_regex", optional=False): t.String(
            max_length=MAX_REGEX_LENGTH
        ),
        t.Key("replace_regex", to_name="replace_regex", optional=False): t.String(
            max_length=MAX_REGEX_LENGTH
        ),
    }
)

guard_intervention_trafaret = t.Dict(
    {
        t.Key("action", to_name="action", optional=False): t.Enum(*GuardAction.ALL),
        t.Key("message", to_name="message", optional=True): t.String(
            max_length=MAX_GUARD_MESSAGE_LENGTH, allow_blank=True
        ),
        t.Key("conditions", to_name="conditions", optional=True): t.Or(
            t.List(
                t.Or(
                    model_guard_intervention_trafaret,
                    modifier_guard_intervention_trafaret,
                ),
                max_length=1,
                min_length=0,
            ),
            t.Null,
        ),
        t.Key("send_notification", to_name="send_notification", optional=True): t.Bool(),
    },
    allow_extra=["*"],
)


guard_metric_baseline_value = t.Dict(
    {
        t.Key("value", to_name="value", optional=False): t.Float(),
    }
)


guard_metric_timestamp_spoofing = t.Dict(
    {
        t.Key("column_name", to_name="column_name", optional=True): t.Or(
            t.String(max_length=MAX_GUARD_COLUMN_NAME_LENGTH), t.Null()
        ),
        t.Key("time_format", to_name="time_format", optional=True): t.Or(
            t.Enum(*DATE_FORMAT_STRINGS), t.Null()
        ),
    }
)


guard_value = t.Dict(
    {
        t.Key("column_name", to_name="column_name", optional=True): t.Or(
            t.String(max_length=MAX_GUARD_COLUMN_NAME_LENGTH), t.Null()
        )
    }
)


custom_metric_trafaret = t.Dict(
    {
        t.Key("name", to_name="name", optional=False): t.String(max_length=MAX_GUARD_NAME_LENGTH),
        t.Key("description", to_name="description", optional=True): t.String(
            max_length=MAX_GUARD_DESCRIPTION_LENGTH
        ),
        t.Key("directionality", to_name="directionality", optional=False): t.Enum(
            *CustomMetricDirectionality.all()
        ),
        t.Key("units", to_name="units", optional=False): t.String(max_length=MAX_GUARD_NAME_LENGTH),
        t.Key("type", to_name="type", optional=False): t.Enum(*CustomMetricAggregationType.all()),
        t.Key("baseline_values", to_name="baseline_values", optional=False): t.List(
            guard_metric_baseline_value,
            max_length=MAX_GUARD_CUSTOM_METRIC_BASELINE_VALUE_LIST_LENGTH,
        ),
        t.Key("timestamp", to_name="timestamp", optional=True): guard_metric_timestamp_spoofing,
        t.Key("value", to_name="value", optional=True): guard_value,
        t.Key("is_model_specific", to_name="is_model_specific", optional=True): t.Bool(),
    }
)


guard_trafaret = t.Dict(
    {
        t.Key("name", to_name="name", optional=False): t.String(max_length=MAX_GUARD_NAME_LENGTH),
        t.Key("description", to_name="description", optional=True): t.String(
            max_length=MAX_GUARD_DESCRIPTION_LENGTH
        ),
        t.Key("type", to_name="type", optional=False): t.Enum(*GuardType.ALL),
        t.Key("stage", to_name="stage", optional=False): t.Or(
            t.List(t.Enum(*GuardStage.ALL)), t.Enum(*GuardStage.ALL)
        ),
        t.Key("llm_type", to_name="llm_type", optional=True): t.Enum(*GuardLLMType.ALL),
        t.Key("ootb_type", to_name="ootb_type", optional=True): t.Enum(*OOTBType.ALL),
        t.Key("deployment_id", to_name="deployment_id", optional=True): t.Or(
            t.String(max_length=OBJECT_ID_LENGTH), t.Null
        ),
        t.Key("model_info", to_name="model_info", optional=True): model_info_trafaret,
        t.Key("intervention", to_name="intervention", optional=True): guard_intervention_trafaret,
        t.Key("custom_metric", to_name="custom_metric", optional=True): t.Or(
            custom_metric_trafaret, t.Null
        ),
        t.Key("openai_api_key", to_name="openai_api_key", optional=True): t.Or(
            t.String(max_length=MAX_TOKEN_LENGTH), t.Null
        ),
        t.Key("openai_deployment_id", to_name="openai_deployment_id", optional=True): t.Or(
            t.String(max_length=OBJECT_ID_LENGTH), t.Null
        ),
        t.Key("openai_api_base", to_name="openai_api_base", optional=True): t.Or(
            t.String(max_length=MAX_URL_LENGTH), t.Null
        ),
        t.Key("faas_url", optional=True): t.Or(t.String(max_length=MAX_URL_LENGTH), t.Null),
        t.Key("copy_citations", optional=True, default=False): t.Bool(),
    },
    allow_extra=["*"],
)


moderation_config_trafaret = t.Dict(
    {
        t.Key(
            "timeout_sec",
            to_name="timeout_sec",
            optional=True,
            default=DEFAULT_GUARD_PREDICTION_TIMEOUT_IN_SEC,
        ): t.Int(gt=1),
        t.Key(
            "timeout_action",
            to_name="timeout_action",
            optional=True,
            default=GuardTimeoutAction.SCORE,
        ): t.Enum(*GuardTimeoutAction.ALL),
        t.Key("guards", to_name="guards", optional=False): t.List(guard_trafaret),
    },
    allow_extra=["*"],
)


basic_credential_trafaret = t.Dict(
    {
        t.Key("credentialType", to_name="credential_type", optional=False): t.Enum("basic"),
        t.Key("password", to_name="password", optional=False): t.String,
    },
    allow_extra=["*"],
)

api_token_credential_trafaret = t.Dict(
    {
        t.Key("credentialType", to_name="credential_type", optional=False): t.Enum("api_token"),
        t.Key("apiToken", to_name="api_token", optional=False): t.String,
    },
    allow_extra=["*"],
)

credential_trafaret = t.Dict(
    {
        t.Key("type", optional=False): t.Enum("credential"),
        t.Key("payload", optional=False): t.Or(
            basic_credential_trafaret, api_token_credential_trafaret
        ),
    }
)


class Guard(ABC):
    def __init__(self, config: dict, stage=None):
        self._name = config["name"]
        self._description = config.get("description")
        self._type = config["type"]
        self._stage = stage if stage else config["stage"]
        self._pipeline = None
        self._model_info = None
        self._custom_metric = None
        self.intervention = None
        self._deployment_id = config.get("deployment_id")
        self._dr_cm = None
        self._faas_url = config.get("faas_url")
        self._copy_citations = config["copy_citations"]

        if config.get("intervention"):
            self.intervention = GuardIntervention(config["intervention"])
        if config.get("custom_metric"):
            self._custom_metric = GuardCustomMetric(config["custom_metric"])
        if config.get("model_info"):
            self._model_info = GuardModelInfo(config["model_info"])

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def type(self) -> GuardType:
        return self._type

    @property
    def stage(self) -> GuardStage:
        return self._stage

    @property
    def faas_url(self) -> str:
        return self._faas_url

    @property
    def copy_citations(self) -> str:
        return self._copy_citations

    def set_pipeline(self, pipeline):
        self._pipeline = pipeline

    @property
    def custom_metric(self):
        return self._custom_metric

    @property
    def llm_type(self):
        return self._llm_type

    @staticmethod
    def get_stage_str(stage):
        return "Prompts" if stage == GuardStage.PROMPT else "Responses"

    def get_latency_custom_metric_name(self):
        return f"{self.name} Guard Latency"

    def get_latency_custom_metric(self):
        return {
            "name": self.get_latency_custom_metric_name(),
            "directionality": CustomMetricDirectionality.LOWER_IS_BETTER,
            "units": "seconds",
            "type": CustomMetricAggregationType.AVERAGE,
            "baselineValue": 0,
            "isModelSpecific": True,
            "timeStep": "hour",
            "description": (
                f"{self.get_latency_custom_metric_name()}. " f" {CUSTOM_METRIC_DESCRIPTION_SUFFIX}"
            ),
        }

    def get_average_score_custom_metric_name(self, stage):
        return f"{self.name} Guard Average Score for {self.get_stage_str(stage)}"

    def get_average_score_metric(self, stage):
        return {
            "name": self.get_average_score_custom_metric_name(stage),
            "directionality": CustomMetricDirectionality.LOWER_IS_BETTER,
            "units": "probability",
            "type": CustomMetricAggregationType.AVERAGE,
            "baselineValue": 0,
            "isModelSpecific": True,
            "timeStep": "hour",
            "description": (
                f"{self.get_average_score_custom_metric_name(stage)}. "
                f" {CUSTOM_METRIC_DESCRIPTION_SUFFIX}"
            ),
        }

    def get_guard_enforced_custom_metric_name(self, stage, moderation_method):
        if moderation_method == GuardAction.REPLACE:
            return f"{self.name} Guard replaced {self.get_stage_str(stage)}"
        return f"{self.name} Guard {moderation_method}ed {self.get_stage_str(stage)}"

    def get_enforced_custom_metric(self, stage, moderation_method):
        return {
            "name": self.get_guard_enforced_custom_metric_name(stage, moderation_method),
            "directionality": CustomMetricDirectionality.LOWER_IS_BETTER,
            "units": "count",
            "type": CustomMetricAggregationType.SUM,
            "baselineValue": 0,
            "isModelSpecific": True,
            "timeStep": "hour",
            "description": (
                f"Number of {self.get_stage_str(stage)} {moderation_method}ed by the "
                f"{self.name} guard.  {CUSTOM_METRIC_DESCRIPTION_SUFFIX}"
            ),
        }

    def get_input_column(self, stage):
        if stage == GuardStage.PROMPT:
            return (
                self._model_info.input_column_name
                if (self._model_info.input_column_name)
                else DEFAULT_PROMPT_COLUMN_NAME
            )
        else:
            return (
                self._model_info.input_column_name
                if (self._model_info and self._model_info.input_column_name)
                else DEFAULT_RESPONSE_COLUMN_NAME
            )

    def get_intervention_action(self):
        if not self.intervention:
            return GuardAction.NONE
        return self.intervention.action

    def build_api_key_env_var_name(self, config, llm_type):
        llm_type_str = ""
        if llm_type == GuardLLMType.AZURE_OPENAI:
            llm_type_str = "AZURE_"

        secret_env_var_name_prefix = f"{SECRET_DEFINITION_PREFIX}_{self._type}_{self._stage}_"
        if self._type == GuardType.NEMO_GUARDRAILS:
            var_name = f"{secret_env_var_name_prefix}{llm_type_str}"
        elif self._type == GuardType.OOTB and config["ootb_type"] == OOTBType.FAITHFULNESS:
            var_name = f"{secret_env_var_name_prefix}{OOTBType.FAITHFULNESS}_{llm_type_str}"
        else:
            raise Exception("Invalid guard config for building env var name")

        var_name += SECRET_DEFINITION_SUFFIX
        return var_name.upper()

    def get_api_key(self, config, llm_type):
        api_key_env_var_name = self.build_api_key_env_var_name(config, llm_type)
        if api_key_env_var_name not in os.environ:
            raise Exception(f"Expected environment variable '{api_key_env_var_name}' not found")

        env_var_value = json.loads(os.environ[api_key_env_var_name])
        credential_config = credential_trafaret.check(env_var_value)
        if credential_config["payload"]["credential_type"] == "basic":
            return credential_config["payload"]["password"]
        else:
            return credential_config["payload"]["api_token"]


class GuardCustomMetric:
    def __init__(self, config):
        self._name = config["name"]
        self._description = config.get("description")
        self._directionality = config["directionality"]
        self._units = config["units"]
        self._type = config["type"]
        self._baseline_values = config["baseline_values"]
        self._timestamp = config.get("timestamp")
        self._value = config.get("value")
        self._is_model_specific = config.get("is_model_specific", False)

    def to_dict(self):
        d = {
            "name": self._name,
            "directionality": self._directionality,
            "units": self._units,
            "type": self._type,
            "baselineValue": self._baseline_values[0]["value"],
            "isModelSpecific": self._is_model_specific,
            "timeStep": "hour",
        }
        if self._description:
            d["description"] = self._description
        return d

    @property
    def name(self):
        return self._name

    @property
    def aggregation_type(self):
        return self._type


class GuardModelInfo:
    def __init__(self, model_config: dict):
        self._model_id = model_config.get("model_id")
        self._input_column_name = model_config["input_column_name"]
        self._target_name = model_config["target_name"]
        self._target_type = model_config["target_type"]
        self._class_names = model_config.get("class_names")
        self.replacement_text_column_name = model_config.get("replacement_text_column_name")

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def input_column_name(self) -> str:
        return self._input_column_name

    @property
    def target_name(self) -> str:
        return self._target_name

    @property
    def target_type(self) -> str:
        return self._target_type

    @property
    def class_names(self):
        return self._class_names


class GuardIntervention:
    def __init__(self, intervention_config: dict) -> None:
        self.action = intervention_config["action"]
        self.message = intervention_config.get("message")
        self.threshold = None
        self.comparator = None
        if (
            "conditions" in intervention_config
            and intervention_config["conditions"] is not None
            and len(intervention_config["conditions"]) > 0
        ):
            self.threshold = intervention_config["conditions"][0].get("comparand")
            self.comparator = intervention_config["conditions"][0].get("comparator")


class PIIGuardIntervention(GuardIntervention):
    def __init__(self, intervention_config: dict):
        super().__init__(intervention_config)
        self._match_regex = intervention_config["conditions"][0]["match_regex"]
        self._replace_regex = intervention_config["conditions"][0]["replace_regex"]

    @property
    def match_regex(self) -> str:
        return self._match_regex

    @property
    def replace_regex(self) -> str:
        return self._replace_regex


class ModelGuard(Guard):
    def __init__(self, config: dict, stage=None):
        super().__init__(config, stage)
        self._deployment_id = config["deployment_id"]
        self._model_info = GuardModelInfo(config["model_info"])
        # dr.Client is set in the Pipeline init, Lets query the deployment
        # to get the prediction server information
        self.deployment = dr.Deployment.get(self._deployment_id)

    @property
    def deployment_id(self) -> str:
        return self._deployment_id

    @property
    def model_info(self):
        return self._model_info

    def get_metric_column_name(self, stage):
        if self.model_info is None:
            raise NotImplementedError("Missing model_info for model guard")
        return self.get_stage_str(stage) + "_" + self._model_info.target_name


class PIIGuard(Guard):
    def __init__(self, config: dict, stage=None):
        super().__init__(config, stage)
        self.intervention = PIIGuardIntervention(config["intervention"])

    def get_metric_column_name(self, stage):
        raise NotImplementedError


class NeMoGuard(Guard):
    def __init__(self, config: dict, stage=None):
        super().__init__(config, stage)
        # NeMo guard only takes a boolean as threshold and equal to as comparator.
        # Threshold bool == TRUE is defined in the colang file as the output of
        # `bot should intervene`
        if self.intervention:
            if not self.intervention.threshold:
                self.intervention.threshold = NEMO_THRESHOLD
            if not self.intervention.comparator:
                self.intervention.comparator = GuardOperatorType.EQUALS

        self.openai_api_base = config.get("openai_api_base")
        self.openai_deployment_id = config.get("openai_deployment_id")

        # Default LLM Type for NeMo is set to OpenAI
        self._llm_type = config.get("llm_type", GuardLLMType.OPENAI)

        self.openai_api_key = self.get_api_key(config, self._llm_type)
        if self.openai_api_key is None:
            raise ValueError("OpenAI API key is required for NeMo Guardrails")

        if self.llm_type == GuardLLMType.OPENAI:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
            llm = None
        elif self.llm_type == GuardLLMType.AZURE_OPENAI:
            if self.openai_api_base is None:
                raise ValueError("Azure OpenAI API base url is required for LLM Guard")
            if self.openai_deployment_id is None:
                raise ValueError("Azure OpenAI deployment ID is required for LLM Guard")
            azure_openai_client = get_azure_openai_client(
                openai_api_key=self.openai_api_key,
                openai_api_base=self.openai_api_base,
                openai_deployment_id=self.openai_deployment_id,
            )
            llm = azure_openai_client
        else:
            raise ValueError(f"Invalid LLMType: {self.llm_type}")

        # Use guard stage to determine whether to read from prompt/response subdirectory
        # for nemo configurations.  "nemo_guardrails" folder is at same level of custom.py
        # So, the config path becomes model_dir + "nemo_guardrails"
        model_dir = os.getcwd()
        nemo_config_path = os.path.join(model_dir, NEMO_GUARDRAILS_DIR)
        self.nemo_rails_config_path = os.path.join(nemo_config_path, self.stage)
        nemo_rails_config = RailsConfig.from_path(config_path=self.nemo_rails_config_path)
        self._nemo_llm_rails = LLMRails(nemo_rails_config, llm=llm)

    def get_metric_column_name(self, stage):
        return self.get_stage_str(stage) + "_" + NEMO_GUARD_COLUMN_NAME

    @property
    def nemo_llm_rails(self):
        return self._nemo_llm_rails


class OOTBGuard(Guard):
    def __init__(self, config: dict, stage=None):
        super().__init__(config, stage)
        self._ootb_type = config["ootb_type"]

    @property
    def ootb_type(self):
        return self._ootb_type

    def get_metric_column_name(self, stage):
        if self._ootb_type == OOTBType.TOKEN_COUNT:
            return self.get_stage_str(stage) + "_" + TOKEN_COUNT_COLUMN_NAME
        elif self._ootb_type == OOTBType.ROUGE_1:
            return self.get_stage_str(stage) + "_" + ROUGE_1_COLUMN_NAME
        elif self._ootb_type == OOTBType.CUSTOM_METRIC:
            return self.get_stage_str(stage) + "_" + self.name
        else:
            raise NotImplementedError(f"No metric column name defined for {self._ootb_type} guard")


class FaithfulnessGuard(OOTBGuard):
    def __init__(self, config: dict, stage=None):
        super().__init__(config, stage)
        self.openai_api_base = config.get("openai_api_base")
        self.openai_deployment_id = config.get("openai_deployment_id")

        if self.stage == GuardStage.PROMPT:
            raise Exception("Faithfulness cannot be configured for the Prompt stage")

        # Default LLM Type for faithfulness is set to AzureOpenAI
        self._llm_type = config.get("llm_type", GuardLLMType.AZURE_OPENAI)

        try:
            self.openai_api_key = self.get_api_key(config, self._llm_type)
        except Exception:
            # Faithfulness is not using credential id yet on some SaaS environments
            self.openai_api_key = config.get("openai_api_key")

        if self.openai_api_key is None:
            raise ValueError("OpenAI API key is required for Faithfulness guard")
        if self.llm_type == GuardLLMType.OPENAI:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
            llm = "default"
        elif self.llm_type == GuardLLMType.AZURE_OPENAI:
            if self.openai_api_base is None:
                raise ValueError("OpenAI API base url is required for LLM Guard")
            if self.openai_deployment_id is None:
                raise ValueError("OpenAI deployment ID is required for LLM Guard")
            azure_openai_client = get_azure_openai_client(
                openai_api_key=self.openai_api_key,
                openai_api_base=self.openai_api_base,
                openai_deployment_id=self.openai_deployment_id,
            )
            llm = azure_openai_client
        else:
            raise ValueError(f"Invalid LLMType: {self.llm_type}")

        service_context = ServiceContext.from_defaults(llm=llm, embed_model=None)
        self._evaluator = FaithfulnessEvaluator(service_context=service_context)

    def get_metric_column_name(self, stage):
        return self.get_stage_str(stage) + "_" + FAITHFULLNESS_COLUMN_NAME

    @property
    def faithfulness_evaluator(self):
        return self._evaluator


class GuardFactory:
    @classmethod
    def _perform_post_validation_checks(cls, guard_config):
        if guard_config["type"] == GuardType.PII:
            return

        if not guard_config.get("intervention"):
            return

        if guard_config["intervention"]["action"] == GuardAction.BLOCK and (
            guard_config["intervention"]["message"] is None
            or len(guard_config["intervention"]["message"]) == 0
        ):
            raise ValueError("Blocked action needs a blocking message")

        if guard_config["intervention"]["action"] == GuardAction.REPLACE:
            if "model_info" not in guard_config:
                raise ValueError("'Replace' action needs model_info section")
            if (
                "replacement_text_column_name" not in guard_config["model_info"]
                or guard_config["model_info"]["replacement_text_column_name"] is None
                or len(guard_config["model_info"]["replacement_text_column_name"]) == 0
            ):
                raise ValueError(
                    "'Replace' action needs valid 'replacement_text_column_name' "
                    "in 'model_info' section of the guard"
                )

        if not guard_config["intervention"].get("conditions"):
            return

        if len(guard_config["intervention"]["conditions"]) == 0:
            return

        condition = guard_config["intervention"]["conditions"][0]
        if condition["comparator"] in GuardOperatorType.REQUIRES_LIST_COMPARAND:
            if not isinstance(condition["comparand"], list):
                raise ValueError(
                    f"Comparand needs to be a list with {condition['comparator']} comparator"
                )
        elif isinstance(condition["comparand"], list):
            raise ValueError(
                f"Comparand needs to be a scalar with {condition['comparator']} comparator"
            )

    @staticmethod
    def create(input_config: dict, stage=None) -> Guard:
        config = guard_trafaret.check(input_config)

        GuardFactory._perform_post_validation_checks(config)

        if config["type"] == GuardType.MODEL:
            guard = ModelGuard(config, stage)
        elif config["type"] == GuardType.PII:
            guard = PIIGuard(config, stage)
        elif config["type"] == GuardType.OOTB:
            if config["ootb_type"] == OOTBType.FAITHFULNESS:
                guard = FaithfulnessGuard(config, stage)
            else:
                guard = OOTBGuard(config, stage)
        elif config["type"] == GuardType.NEMO_GUARDRAILS:
            guard = NeMoGuard(config, stage)
        else:
            raise ValueError(f'Invalid guard type: {config["type"]}')

        return guard
