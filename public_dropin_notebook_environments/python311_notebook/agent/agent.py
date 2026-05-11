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


@app.websocket_route("/ssh")
async def ssh_endpoint(websocket: WebSocket) -> None:
    """Bridge a WebSocket connection to the local sshd (port 8022)."""
    await websocket.accept()
    try:
        reader, writer = await asyncio.open_connection('127.0.0.1', 8022)
    except OSError as exc:
        logger.error("Failed to connect to sshd: %s", exc)
        await websocket.close(code=1011, reason=str(exc))
        return

    async def ws_to_tcp() -> None:
        try:
            while True:
                data = await websocket.receive_bytes()
                writer.write(data)
                await writer.drain()
        except Exception:
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def tcp_to_ws() -> None:
        try:
            while True:
                data = await reader.read(4096)
                if not data:
                    break
                await websocket.send_bytes(data)
        except Exception:
            pass

    tasks = [asyncio.create_task(ws_to_tcp()), asyncio.create_task(tcp_to_ws())]
    _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    try:
        await websocket.close()
    except Exception:
        pass


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
