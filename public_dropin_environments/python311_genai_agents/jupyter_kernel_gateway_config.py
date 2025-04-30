# Copyright 2022 DataRobot, Inc. and its affiliates.
# All rights reserved.
# DataRobot, Inc. Confidential.
# This is unpublished proprietary source code of DataRobot, Inc.
# and its affiliates.
# The copyright notice above does not evidence any actual or intended
# publication of such source code.

c.KernelGatewayApp.ip = "0.0.0.0"  # nosec
c.KernelGatewayApp.prespawn_count = 1
c.KernelGatewayApp.max_kernels = 100
c.KernelGatewayApp.default_kernel_name = "python3"
c.JupyterWebsocketPersonality.list_kernels = True
c.KernelRestarter.restart_limit = (
    3  # if restart happens 3 consecutive times (before kernel is ready)
)

c.KernelGatewayApp.logging_config = {
    "formatters": {
        "console": {"class": "ecs_logging.StdlibFormatter"},
    },
    "loggers": {
        "KernelGatewayApp": {
            "handlers": ["console"],
        }
    },
}
