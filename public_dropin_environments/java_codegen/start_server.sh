#!/bin/sh
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
echo "Starting Custom Model environment with DRUM prediction server"

if [ "${ENABLE_CUSTOM_MODEL_RUNTIME_ENV_DUMP}" = 1 ]; then
    echo "Environment variables:"
    env
fi

# FIPS 140-3 (RAPTOR-21063): with DATAROBOT_FIPS_ENFORCE=true, run the JVM FIPS self-test before
# the server starts, so a failure stops the container (exit 78) instead of leaving DRUM retrying a
# Java gateway that never came up. The self-test itself reads DATAROBOT_FIPS_ENFORCE.
case "${DATAROBOT_FIPS_ENFORCE:-}" in
    *[Tt][Rr][Uu][Ee]*)
        java --add-modules org.bouncycastle.fips.core /usr/share/java/datarobot-fips/FipsSelfTest.java || exit $?
        ;;
esac

echo
echo "Executing command: drum server $*"
echo
exec drum server "$@"
