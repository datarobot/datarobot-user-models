#!/bin/sh
# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

# FIPS power-on self-test (openssl-fips-test) before anything else starts; a no-op unless DATAROBOT_FIPS_ENFORCE=true.
sh /etc/datarobot-fips/fips_selftest_entrypoint.sh true || exit $?
echo "Starting Custom Model environment with DRUM prediction server"

if [ "${ENABLE_CUSTOM_MODEL_RUNTIME_ENV_DUMP}" = 1 ]; then
    echo "Environment variables:"
    env
fi

echo
echo "Executing command: drum server $*"
echo
exec drum server "$@"
