"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
from datetime import datetime, timezone

import requests
from requests import HTTPError

from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX


class MLOpsStatusReporter:
    def __init__(
        self,
        mlops_service_url,
        mlops_api_token,
        deployment_id,
        verify_ssl=True,
        total_deployment_stages=None,
    ):
        self.logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)
        self.mlops_service_url = mlops_service_url
        self.mlops_api_token = mlops_api_token
        self.verify_ssl = verify_ssl
        self.deployment_id = deployment_id
        self.total_deployment_stages = total_deployment_stages
        self.current_stage = 0

    def report_deployment(self, message: str):
        if not all([self.deployment_id, self.mlops_service_url, self.mlops_api_token]):
            # custom model testing has no deployment_id, skip reporting
            return

        self.logger.info(message)
        remote_events_url = f"{self.mlops_service_url}/remoteEvents/"

        title = "Deployment status"
        if self.total_deployment_stages:
            self.current_stage += 1
            title = f"Deployment stage {self.current_stage} out of {self.total_deployment_stages}"

        auth_header = {"Authorization": f"Bearer {self.mlops_api_token}"}
        event_payload = {
            "eventType": "deploymentInfo",
            "title": title,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "deploymentId": self.deployment_id,
        }

        try:
            response = requests.post(
                url=remote_events_url,
                json=event_payload,
                headers=auth_header,
                verify=self.verify_ssl,
                timeout=5,
            )
            response.raise_for_status()
        except (ConnectionError, HTTPError):
            self.logger.warning("Deployment event can not be reported to MLOPS", exc_info=True)
