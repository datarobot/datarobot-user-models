#!/bin/bash

export HOME="/home/notebooks"

# setup the working directory for the kernel
if [ -z "$1" ]; then
    # Set default working directory if no argument is provided
    WORKING_DIR="/home/notebooks"
else
    # Use the provided working directory
    WORKING_DIR="$1"
fi

export WORKING_DIR

# FLEET-8918: a hardened cluster can be made to run this session at a uid the image was not
# built with, and then $HOME and every image layer are read-only, so the set-up below fails
# and the session comes up subtly broken while the pod still reports 1/1 Running.
#
# Every redirect here is gated on $HOME actually being unwritable, so a default install is
# untouched: same paths, same files, same PYTHONPATH as before.
#
# /var/run/notebooks cannot serve as the writable location: it is root-owned at runtime,
# created by the ssh secret mount, so writes there are silently lost.
export NOTEBOOKS_RW_DIR="${NOTEBOOKS_RW_DIR:-/home/notebooks/.nbx-rw}"
export IPYTHONDIR="${HOME}/.ipython"

if [ -w "$HOME" ]; then
    NBX_HOME_WRITABLE=true
else
    NBX_HOME_WRITABLE=false
    mkdir -p "${NOTEBOOKS_RW_DIR}/profile.d" || \
        echo "WARNING: neither ${HOME} nor ${NOTEBOOKS_RW_DIR} is writable - session set-up will be degraded." >&2

    # git >= 2.32, IPython >= 7 and Jupyter all read these, so none of them needs a writable $HOME.
    export GIT_CONFIG_GLOBAL="${NOTEBOOKS_RW_DIR}/gitconfig"
    export IPYTHONDIR="${NOTEBOOKS_RW_DIR}/ipython"
    export JUPYTER_CONFIG_DIR="${NOTEBOOKS_RW_DIR}/jupyter"
    export JUPYTER_DATA_DIR="${NOTEBOOKS_RW_DIR}/jupyter/data"
    export JUPYTER_RUNTIME_DIR="${NOTEBOOKS_RW_DIR}/jupyter/runtime"
    mkdir -p "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" 2>/dev/null || true
    touch "$GIT_CONFIG_GLOBAL" 2>/dev/null || true

    # The image bakes PYTHONPATH=/home/notebooks/.ipython/extensions, which is where the
    # ipython config is copied further down. With IPYTHONDIR moved off it, dataframe_formatter
    # stops importing and dataframe rendering silently disappears.
    export PYTHONPATH="${IPYTHONDIR}/extensions:${PYTHONPATH}"
fi
export NBX_HOME_WRITABLE

VERBOSE_MODE=true
# shellcheck disable=SC1091
source /etc/system/kernel/setup-venv.sh $VERBOSE_MODE

cd /etc/system/kernel/agent || exit
nohup uvicorn agent:app --host 0.0.0.0 --port 8889 &

# shellcheck disable=SC1091
source /etc/system/kernel/common-user-limits.sh

# shellcheck disable=SC1091
source /etc/system/kernel/setup-ssh.sh
cp -L /var/run/notebooks/ssh/authorized_keys/notebooks /etc/authorized_keys/ && chmod 600 /etc/authorized_keys/notebooks
# mkdir -p, not mkdir: when nbx-operator mounts an emptyDir over this path the directory
# already exists, and a plain mkdir fails with "File exists" so the cp never runs and sshd
# comes up with no host key - while the pod still reports 1/1 Running.
mkdir -p /etc/ssh/keys && cp -L /var/run/notebooks/ssh/keys/ssh_host_* /etc/ssh/keys/ && chmod 600 /etc/ssh/keys/ssh_host_*
nohup /usr/sbin/sshd -D &

# Ensure proper permissions on the directory used by cache daemon to ensure it starts (create dir. if needed)
if mkdir -p /home/notebooks/storage/.cache/git/credential 2>/dev/null; then
    chmod 700 /home/notebooks/storage/.cache/git/credential
else
    echo "WARNING: /home/notebooks/storage is not writable, so the git credential cache daemon may not start." >&2
fi
# Initialize the git helper. Features are turned on/off dependent on `GITHELPER_*` env vars
# drgithelper writes two dotfiles into $HOME and has no flag to move them, so an unwritable
# $HOME gets this one call a writable one. A default install keeps $HOME, and with it the
# git credential cache in its usual place and surviving a session restart.
if [ "$NBX_HOME_WRITABLE" = true ]; then
    /etc/system/kernel/drgithelper configs set
else
    NBX_TOOL_HOME="${NOTEBOOKS_RW_DIR}/home"
    mkdir -p "$NBX_TOOL_HOME" 2>/dev/null || true
    HOME="$NBX_TOOL_HOME" /etc/system/kernel/drgithelper configs set
fi

# no trailing slash in the working dir path
git config --global --add safe.directory "${WORKING_DIR%/}"

# setup the working directory for the kernel
cd "$WORKING_DIR" || exit

# setup ipython extensions
# `/etc/ipython/.` rather than `/etc/ipython/`: the trailing-slash form nests a second
# ipython/ directory under the target on any re-run.
mkdir -p "${IPYTHONDIR}" && cp -r /etc/ipython/. "${IPYTHONDIR}/"

# clear out kubernetes_specific env vars before starting kernel gateway as it will inherit them
prefix="KUBERNETES_"; for var in $(printenv | cut -d= -f1); do [[ "$var" == "$prefix"* ]] && unset "$var"; done

exec jupyter kernelgateway --config=/etc/system/kernel/jupyter_kernel_gateway_config.py --debug
