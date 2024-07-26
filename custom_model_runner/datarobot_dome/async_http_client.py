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
import asyncio
import atexit
import logging
import os
from io import StringIO

import aiohttp
import backoff
import nest_asyncio
import pandas as pd

from datarobot_dome.constants import DATAROBOT_SERVERLESS_PLATFORM
from datarobot_dome.constants import DEFAULT_GUARD_PREDICTION_TIMEOUT_IN_SEC
from datarobot_dome.constants import LOGGER_NAME_PREFIX
from datarobot_dome.constants import RETRY_COUNT

RETRY_STATUS_CODES = [413, 502, 504]
RETRY_AFTER_STATUS_CODES = [429, 503]


# We want this logger to be available for backoff too, hence defining outside the class
logger = logging.getLogger(LOGGER_NAME_PREFIX + ".AsyncHTTPClient")


# Event handlers for backoff
def _timeout_backoff_handler(details):
    logger.warning(
        f"HTTP Timeout: Backing off {details['wait']} seconds after {details['tries']} tries"
    )


def _timeout_giveup_handler(details):
    url = details["args"][1]
    logger.error(f"Giving up predicting on {url}, Retried {details['tries']} after HTTP Timeout")


def _retry_backoff_handler(details):
    status_code = details["value"].status
    message = details["value"].reason
    retry_after_value = details["value"].headers.get("Retry-After")
    logger.warning(
        f"Received status code {status_code}, message {message},"
        f" Retry-After val: {retry_after_value} "
        f"Backing off {details['wait']} seconds after {details['tries']} tries"
    )


def _retry_giveup_handler(details):
    message = (
        f"Giving up predicting on {details['args'][1]}, Retried {details['tries']} retries, "
        f"elapsed time {details['elapsed']} sec, but couldn't get predictions"
    )
    raise Exception(message)


class AsyncHTTPClient:
    def __init__(self, timeout=DEFAULT_GUARD_PREDICTION_TIMEOUT_IN_SEC):
        self.headers = {
            "Content-Type": "text/csv",
            "Accept": "text/csv",
            "Authorization": f"Bearer {os.environ['DATAROBOT_API_TOKEN']}",
        }
        self.session = None
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError as e:
            if str(e).startswith("There is no current event loop in thread") or str(e).startswith(
                "Event loop is closed"
            ):
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
            else:
                raise
        self.loop.run_until_complete(self.__create_client_session(timeout))
        self.loop.set_debug(True)
        nest_asyncio.apply(loop=self.loop)

        atexit.register(self.shutdown)

    async def __create_client_session(self, timeout):
        client_timeout = aiohttp.ClientTimeout(
            connect=timeout, sock_connect=timeout, sock_read=timeout
        )
        # Creation of client session needs to happen within in async function
        self.session = aiohttp.ClientSession(timeout=client_timeout)

    def shutdown(self):
        asyncio.run(self.session.close())

    @staticmethod
    def is_serverless_deployment(deployment):
        if not deployment.prediction_environment:
            return False

        if deployment.prediction_environment.get("platform") == DATAROBOT_SERVERLESS_PLATFORM:
            return True

        return False

    @backoff.on_predicate(
        backoff.runtime,
        predicate=lambda r: r.status in RETRY_AFTER_STATUS_CODES,
        value=lambda r: int(r.headers.get("Retry-After", DEFAULT_GUARD_PREDICTION_TIMEOUT_IN_SEC)),
        max_time=RETRY_COUNT * DEFAULT_GUARD_PREDICTION_TIMEOUT_IN_SEC,
        max_tries=RETRY_COUNT,
        jitter=None,
        logger=logger,
        on_backoff=_retry_backoff_handler,
        on_giveup=_retry_giveup_handler,
    )
    @backoff.on_predicate(
        backoff.fibo,
        predicate=lambda r: r.status in RETRY_STATUS_CODES,
        jitter=None,
        max_tries=RETRY_COUNT,
        max_time=RETRY_COUNT * DEFAULT_GUARD_PREDICTION_TIMEOUT_IN_SEC,
        logger=logger,
        on_backoff=_retry_backoff_handler,
        on_giveup=_retry_giveup_handler,
    )
    @backoff.on_exception(
        backoff.fibo,
        asyncio.TimeoutError,
        max_tries=RETRY_COUNT,
        max_time=RETRY_COUNT * DEFAULT_GUARD_PREDICTION_TIMEOUT_IN_SEC,
        logger=logger,
        on_backoff=_timeout_backoff_handler,
        on_giveup=_timeout_giveup_handler,
        raise_on_giveup=True,
    )
    async def post_predict_request(self, url_path, input_df):
        return await self.session.post(
            url_path, data=input_df.to_csv(index=False), headers=self.headers
        )

    async def predict(self, deployment, input_df):
        deployment_id = str(deployment.id)
        if self.is_serverless_deployment(deployment):
            url_path = f"{os.environ['DATAROBOT_ENDPOINT']}"
        else:
            prediction_server = deployment.default_prediction_server
            if not prediction_server:
                raise ValueError(
                    "Can't make prediction request because Deployment object doesn't contain "
                    "default prediction server"
                )
            datarobot_key = prediction_server.get("datarobot-key")
            if datarobot_key:
                self.headers["datarobot-key"] = datarobot_key

            url_path = f"{prediction_server['url']}/predApi/v1.0"

        url_path += f"/deployments/{deployment_id}/predictions"
        response = await self.post_predict_request(url_path, input_df)
        csv_data = await response.text()
        return pd.read_csv(StringIO(csv_data))
