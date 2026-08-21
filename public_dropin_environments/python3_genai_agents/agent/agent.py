# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
from typing import Union

import ecs_logging
from cgroup_watchers import CGroupFileReader
from cgroup_watchers import CGroupV2FileReader
from cgroup_watchers import CGroupVersionUnsupported
from cgroup_watchers import CGroupWatcher
from cgroup_watchers import DummyWatcher
from cgroup_watchers import SystemWatcher
from fastapi import FastAPI
from fastapi import WebSocket
from websockets.exceptions import ConnectionClosedError
from websockets.exceptions import ConnectionClosedOK

logger = logging.getLogger("kernel_agent")

logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(ecs_logging.StdlibFormatter())
logger.addHandler(handler)

app = FastAPI()

watcher: Union[CGroupWatcher, DummyWatcher]

try:
    watcher = CGroupWatcher(CGroupV2FileReader(), SystemWatcher())
    logger.info("Using CGroup V2 Watcher")
except CGroupVersionUnsupported:
    logger.debug("CGroup V2 not available, trying V1")
    try:
        watcher = CGroupWatcher(CGroupFileReader(), SystemWatcher())
        logger.info("Using CGroup V1 Watcher")
    except CGroupVersionUnsupported:
        logger.debug("CGroup V1 not available")
        logger.warning(
            "CGroup not supported. Using DummyWatcher with SystemWatcher (psutil) "
            "for system-wide resource metrics instead of container-specific cgroup metrics"
        )
        watcher = DummyWatcher()


@app.websocket_route("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            await websocket.send_json(
                {
                    "cpu_percent": watcher.cpu_usage_percentage(),
                    "mem_percent": watcher.memory_usage_percentage(),
                }
            )

            await asyncio.sleep(3)
    except ConnectionClosedError:
        logger.warning(
            "utilization consumer unconnected",
            extra={"connection": websocket.client},
            exc_info=True,
        )
    except ConnectionClosedOK:
        # https://github.com/encode/starlette/issues/759
        logger.info("utilization consumer unconnected", extra={"connection": websocket.client})
