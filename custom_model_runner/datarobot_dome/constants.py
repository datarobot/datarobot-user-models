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
import os
from enum import Enum

__GUARD_ASSOCIATION_IDS_COLUMN_NAME__ = "datarobot_guard_association_id"

LOGGER_NAME_PREFIX = "moderations"

DEFAULT_PROMPT_COLUMN_NAME = "promptText"
DEFAULT_RESPONSE_COLUMN_NAME = "completion"

NEMO_GUARDRAILS_DIR = "nemo_guardrails"


TOKEN_COUNT_COLUMN_NAME = "token_count"
ROUGE_1_COLUMN_NAME = "rouge_1"
NEMO_GUARD_COLUMN_NAME = "nemo_guard"
FAITHFULLNESS_COLUMN_NAME = "faithfulness"

CUSTOM_METRIC_DESCRIPTION_SUFFIX = "Created by DataRobot Moderation System"

# Setting timeout at 10 sec, we have 5 retries, so we approximately wait for
# 50 sec, before giving up on the guard.
DEFAULT_GUARD_PREDICTION_TIMEOUT_IN_SEC = 10

# Connect and read retries count
RETRY_COUNT = 5

DEFAULT_GUARD_CONFIG_FILE = os.path.join(".", "moderation_config.yaml")
DATAROBOT_SERVERLESS_PLATFORM = "datarobotServerless"

SECRET_DEFINITION_PREFIX = "MLOPS_RUNTIME_PARAM_MODERATION"
SECRET_DEFINITION_SUFFIX = "OPENAI_API_KEY"


class GuardType:
    OOTB = "ootb"  # Out of the Box
    MODEL = "model"  # wraps a deployed model
    NEMO_GUARDRAILS = "nemo_guardrails"  # Nemo guardrails
    PII = "pii"  # Internal PII detection

    ALL = [MODEL, NEMO_GUARDRAILS, PII, OOTB]


class OOTBType:
    TOKEN_COUNT = "token_count"
    ROUGE_1 = "rouge_1"
    FAITHFULNESS = "faithfulness"
    CUSTOM_METRIC = "custom_metric"

    ALL = [TOKEN_COUNT, ROUGE_1, FAITHFULNESS, CUSTOM_METRIC]


class GuardStage:
    """When can this guard operate?"""

    PROMPT = "prompt"
    RESPONSE = "response"

    ALL = [PROMPT, RESPONSE]


class GuardExecutionLocation(Enum):
    LOCAL = "local"
    REMOTE = "remote"

    ALL = [LOCAL, REMOTE]


class GuardAction:
    """
    Defines actions a guard can take.
    All guards report their decisions; 'report' means do nothing else.
    """

    BLOCK = "block"
    REPORT = "report"
    REPLACE = "replace"
    NONE = None

    ALL = [BLOCK, REPORT, REPLACE, NONE]


class GuardModelTargetType:
    """
    Guards support a subset of DataRobot model target types.
    Ref: common.engine.TargetType
    """

    BINARY = "Binary"
    REGRESSION = "Regression"
    MULTICLASS = "Multiclass"
    TEXT_GENERATION = "TextGeneration"

    ALL = [BINARY, REGRESSION, MULTICLASS, TEXT_GENERATION]


class GuardOperatorType:
    """
    Defines what the guard should do with the metric or prediction.
    Applies when the parameter type is OPERATOR.
    Typically this compares against a number, or looks for a matching category or string.
    """

    GREATER_THAN = "greaterThan"
    LESS_THAN = "lessThan"
    EQUALS = "equals"
    NOT_EQUALS = "notEquals"
    IS = "is"
    IS_NOT = "isNot"
    MATCHES = "matches"
    DOES_NOT_MATCH = "doesNotMatch"
    CONTAINS = "contains"
    DOES_NOT_CONTAIN = "doesNotContain"

    ALL = [
        GREATER_THAN,
        LESS_THAN,
        EQUALS,
        NOT_EQUALS,
        IS,
        IS_NOT,
        MATCHES,
        DOES_NOT_MATCH,
        CONTAINS,
        DOES_NOT_CONTAIN,
    ]

    REQUIRES_LIST_COMPARAND = [MATCHES, DOES_NOT_MATCH, CONTAINS, DOES_NOT_CONTAIN]


class GuardLLMType:
    """LLM Types to use for guards"""

    OPENAI = "openAi"
    AZURE_OPENAI = "azureOpenAi"

    ALL = [OPENAI, AZURE_OPENAI]


class GuardTimeoutAction:
    """Actions if the guard times out"""

    # Block the prompt / response if the guard times out
    BLOCK = "block"

    # Continue scoring prompt / returning response if the
    # guard times out
    SCORE = "score"

    ALL = [BLOCK, SCORE]


DATE_FORMAT_STRINGS = [
    "%m/%d/%Y",
    "%m/%d/%y",
    "%d/%m/%y",
    "%m-%d-%Y",
    "%m-%d-%y",
    "%Y/%m/%d",
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y.%m.%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M",
    "%y/%m/%d",
    "%y-%m-%d",
    "%y-%m-%d %H:%M:%S",
    "%y.%m.%d %H:%M:%S",
    "%y/%m/%d %H:%M:%S",
    "%y-%m-%d %H:%M",
    "%y.%m.%d %H:%M",
    "%y/%m/%d %H:%M",
    "%m/%d/%Y %H:%M",
    "%m/%d/%y %H:%M",
    "%d/%m/%Y %H:%M",
    "%d/%m/%y %H:%M",
    "%m-%d-%Y %H:%M",
    "%m-%d-%y %H:%M",
    "%d-%m-%Y %H:%M",
    "%d-%m-%y %H:%M",
    "%m.%d.%Y %H:%M",
    "%m/%d.%y %H:%M",
    "%d.%m.%Y %H:%M",
    "%d.%m.%y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%y %H:%M:%S",
    "%m-%d-%Y %H:%M:%S",
    "%m-%d-%y %H:%M:%S",
    "%m.%d.%Y %H:%M:%S",
    "%m.%d.%y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%y-%m-%dT%H:%M:%SZ",
    "%Y.%m.%d %H:%M:%S.%f",
    "%y.%m.%d %H:%M:%S.%f",
    "%Y.%m.%dT%H:%M:%S.%fZ",
    "%y.%m.%dT%H:%M:%S.%fZ",
    "%Y.%m.%dT%H:%M:%S.%f",
    "%y.%m.%dT%H:%M:%S.%f",
    "%Y.%m.%dT%H:%M:%S",
    "%y.%m.%dT%H:%M:%S",
    "%Y.%m.%dT%H:%M:%SZ",
    "%y.%m.%dT%H:%M:%SZ",
    "%Y%m%d",
    "%m %d %Y %H %M %S",
    "%m %d %y %H %M %S",
    "%H:%M",
    "%M:%S",
    "%H:%M:%S",
    "%Y %m %d %H %M %S",
    "%y %m %d %H %M %S",
    "%Y %m %d",
    "%y %m %d",
    "%d/%m/%Y",
    "%Y-%d-%m",
    "%y-%d-%m",
    "%Y/%d/%m %H:%M:%S.%f",
    "%Y/%d/%m %H:%M:%S.%fZ",
    "%Y/%m/%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S.%fZ",
    "%y/%d/%m %H:%M:%S.%f",
    "%y/%d/%m %H:%M:%S.%fZ",
    "%y/%m/%d %H:%M:%S.%f",
    "%y/%m/%d %H:%M:%S.%fZ",
    "%m.%d.%Y",
    "%m.%d.%y",
    "%d.%m.%y",
    "%d.%m.%Y",
    "%Y.%m.%d",
    "%Y.%d.%m",
    "%y.%m.%d",
    "%y.%d.%m",
]
