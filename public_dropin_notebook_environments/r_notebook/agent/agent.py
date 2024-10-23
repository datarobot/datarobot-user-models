# Copyright 2022 DataRobot, Inc. and its affiliates.
# All rights reserved.
# DataRobot, Inc. Confidential.
# This is unpublished proprietary source code of DataRobot, Inc.
# and its affiliates.
# The copyright notice above does not evidence any actual or intended
# publication of such source code.

import asyncio

from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from cgroup_watchers import (
    CGroupFileReader,
    CGroupWatcher,
    DummyWatcher,
    SystemWatcher,
    CGroupVersionUnsupported,
)
from fastapi import FastAPI, WebSocket
import logging
import ecs_logging

logger = logging.getLogger("kernel_agent")

logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(ecs_logging.StdlibFormatter())
logger.addHandler(handler)

app = FastAPI()

try:
    watcher = CGroupWatcher(CGroupFileReader(), SystemWatcher())
except CGroupVersionUnsupported:
    logger.warning("CGroup Version Unsupported. Dummy utilization will be broadcasted")
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
