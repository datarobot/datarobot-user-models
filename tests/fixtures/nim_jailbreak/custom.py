#
# Copyright 2025 DataRobot, Inc. and its affiliates.
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
import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def score(data, model: str, base_url: str, **kwargs):
    prompts = data["text"]
    jailbreak_scores = []

    for prompt in prompts:
        if prompt is None:
            jailbreak_scores.append(0)
            continue
        data = {"input": prompt}
        response = requests.post(f"{base_url}/v1/classify", json=data)
        result_dict = response.json()
        score = 1 if result_dict["jailbreak"] else 0
        jailbreak_scores.append(score)

    return pd.DataFrame(
        {
            kwargs["negative_class_label"]: [1 - score for score in jailbreak_scores],
            kwargs["positive_class_label"]: jailbreak_scores,
        }
    )
