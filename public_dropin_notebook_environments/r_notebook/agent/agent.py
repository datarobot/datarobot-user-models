# Copyright 2022 DataRobot, Inc. and its affiliates.
# All rights reserved.
# DataRobot, Inc. Confidential.
# This is unpublished proprietary source code of DataRobot, Inc.
# and its affiliates.
# The copyright notice above does not evidence any actual or intended
# publication of such source code.

import asyncio

from cgroup_monitor import CGroupFileReader, CGroupWatcher, SystemWatcher
from fastapi import FastAPI, WebSocket

app = FastAPI()


@app.websocket_route("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cgroup_monitor = CGroupWatcher(CGroupFileReader(), SystemWatcher())
    while True:
        await websocket.send_json(
            {
                'cpu_percent': cgroup_monitor.cpu_usage_percentage(),
                'mem_percent': cgroup_monitor.memory_usage_percentage(),
            }
        )

        await asyncio.sleep(3)
