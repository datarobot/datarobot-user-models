#!/bin/bash
# FLEET-8918: sources the session env file that start_server.sh may write to a volume when
# /etc/profile.d is not writable, treating both the directory and the file as optional so a
# container without the volume still gets a working shell.
_nbx_rw_env="${NOTEBOOKS_RW_DIR:-/home/notebooks/.nbx-rw}/profile.d/notebooks-load-env.sh"
if [ -r "$_nbx_rw_env" ]; then
    # shellcheck disable=SC1090
    . "$_nbx_rw_env"
fi
unset _nbx_rw_env
