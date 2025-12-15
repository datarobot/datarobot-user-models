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
mkdir /etc/ssh/keys && cp -L /var/run/notebooks/ssh/keys/ssh_host_* /etc/ssh/keys/ && chmod 600 /etc/ssh/keys/ssh_host_*
nohup /usr/sbin/sshd -D &

# no trailing slash in the working dir path
git config --global --add safe.directory "${WORKING_DIR%/}"

# setup the working directory for the kernel
cd "$WORKING_DIR" || exit

# setup ipython extensions
cp -r /etc/ipython/ /home/notebooks/.ipython/

# Copy agent runtime from work directory to executable directory
cp /etc/system/kernel/run_agent.py /home/notebooks/storage/run_agent.py

# clear out kubernetes_specific env vars before starting kernel gateway as it will inherit them
prefix="KUBERNETES_"; for var in $(printenv | cut -d= -f1); do [[ "$var" == "$prefix"* ]] && unset "$var"; done

exec jupyter kernelgateway --config=/etc/system/kernel/jupyter_kernel_gateway_config.py --debug
