#
#  Copyright 2026 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import socket

from gevent.socket import wait_read
from gunicorn.workers.ggevent import GeventWorker


class DrumGeventWorker(GeventWorker):
    """Gevent worker that drops connections which never send a request.

    With keepalive=0 gunicorn reads the first request without any timeout, so a
    connection that is opened but never used (e.g. one Go's http.Transport dialed
    for a request that was cancelled meanwhile, then parked in its idle pool for
    up to 90s) holds one of the worker_connections slots until the client closes it.
    """

    first_byte_timeout = 2

    def handle(self, listener, client, addr):
        if self.first_byte_timeout:
            try:
                wait_read(client.fileno(), timeout=self.first_byte_timeout)
            except socket.timeout:
                self.log.debug(
                    "Closing connection from %s: no data within %ss", addr, self.first_byte_timeout
                )
                client.close()
                return
        super().handle(listener, client, addr)
