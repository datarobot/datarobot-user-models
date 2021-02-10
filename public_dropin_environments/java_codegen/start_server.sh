#!/bin/sh
echo "Starting Custom Model environment with DRUM prediction server"
echo "Environment variables:"
env
echo

CMD="drum server $@"
echo "Executing command: ${CMD}"
echo
exec ${CMD}
