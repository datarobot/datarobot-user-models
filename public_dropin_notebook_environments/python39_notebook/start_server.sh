#!/bin/bash

if [ -z "$1" ]; then
    # Set default working directory if no argument is provided
    WORKING_DIR="/home/notebooks"
else
    # Use the provided working directory
    WORKING_DIR="$1"
fi

export WORKING_DIR

cd /etc/system/kernel/agent || exit
# shellcheck disable=SC1091
source /etc/system/kernel/agent/.venv/bin/activate
nohup uvicorn agent:app --host 0.0.0.0 --port 8889 &

# shellcheck disable=SC1091
source /etc/system/kernel/setup-ssh.sh
source /etc/system/kernel/common-user-limits.sh
cp -L /var/run/notebooks/ssh/authorized_keys/notebooks /etc/authorized_keys/ && chmod 600 /etc/authorized_keys/notebooks
mkdir /etc/ssh/keys && cp -L /var/run/notebooks/ssh/keys/ssh_host_* /etc/ssh/keys/ && chmod 600 /etc/ssh/keys/ssh_host_*
nohup /usr/sbin/sshd -D &

cd /etc/system/kernel || exit
# shellcheck disable=SC1091
source /etc/system/kernel/.venv/bin/activate

# setup the working directory for the kernel
cd "$WORKING_DIR" || exit

exec jupyter kernelgateway --config=/etc/system/kernel/jupyter_kernel_gateway_config.py --debug
