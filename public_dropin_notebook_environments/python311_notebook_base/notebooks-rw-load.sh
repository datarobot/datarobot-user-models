#!/bin/bash
# FLEET-8918: sources the session env file start_server.sh generates onto a volume, since
# /etc/profile.d is writable only by the uid the image was built with.
#
# Directory and file are both optional: a container with no volume, or a login shell opened
# before start_server.sh has run, still gets a working shell.
_nbx_rw_env="${NOTEBOOKS_RW_DIR:-/home/notebooks/.nbx-rw}/profile.d/notebooks-load-env.sh"
if [ -r "$_nbx_rw_env" ]; then
    # shellcheck disable=SC1090
    . "$_nbx_rw_env"
fi
unset _nbx_rw_env
