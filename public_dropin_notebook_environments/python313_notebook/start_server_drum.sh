#!/bin/sh
# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

# FIPS 140-3 startup self-test (CFX-8623): halts with exit 78 before anything starts unless FIPS crypto is in effect.
sh /etc/datarobot-fips/fips_selftest.sh || exit $?

echo "Starting Custom Model environment with DRUM prediction server"

if [ "${ENABLE_CUSTOM_MODEL_RUNTIME_ENV_DUMP}" = 1 ]; then
    echo "Environment variables:"
    env
fi

echo
echo "Executing command: drum server $*"
echo
exec drum server "$@"
