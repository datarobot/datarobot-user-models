#!/bin/sh
echo "Environment variables:"
env

CMD="drum server"
echo "Executing: ${CMD}"
exec ${CMD}
