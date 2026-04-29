# Copyright 2024 DataRobot, Inc. and its affiliates.
# All rights reserved.
# DataRobot, Inc. Confidential.
# This is unpublished proprietary source code of DataRobot, Inc.
# and its affiliates.
# The copyright notice above does not evidence any actual or intended
# publication of such source code.

import asyncio
import logging
from typing import Union

import ecs_logging
from cgroup_watchers import (
    CGroupFileReader,
    CGroupV2FileReader,
    CGroupVersionUnsupported,
    CGroupWatcher,
    DummyWatcher,
    SystemWatcher,
)
from fastapi import FastAPI, WebSocket
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

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


@app.websocket_route("/ws")  # type: ignore[untyped-decorator]
async def websocket_endpoint(websocket: WebSocket) -> None:
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
