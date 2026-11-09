#!/bin/bash

# FIPS power-on self-test (openssl-fips-test) before anything else starts; a no-op unless DATAROBOT_FIPS_ENFORCE=true.
sh /etc/datarobot-fips/fips_selftest_entrypoint.sh true || exit $?

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

# fips/sitecustomize.py sets GODEBUG and DATAROBOT_FIPS_GUARD_LOGGED inside Python processes only; exported here
# so the sshd env snapshot in setup-ssh.sh carries them into terminals.
eval "$(python -c 'import os, shlex
for k in ("GODEBUG", "DATAROBOT_FIPS_GUARD_LOGGED"):
    if k in os.environ: print(f"export {k}={shlex.quote(os.environ[k])}")')"

# shellcheck disable=SC1091
source /etc/system/kernel/setup-ssh.sh
cp -L /var/run/notebooks/ssh/authorized_keys/notebooks /etc/authorized_keys/ && chmod 600 /etc/authorized_keys/notebooks
mkdir /etc/ssh/keys && cp -L /var/run/notebooks/ssh/keys/ssh_host_* /etc/ssh/keys/ && chmod 600 /etc/ssh/keys/ssh_host_*
nohup /usr/sbin/sshd -D &

# setup the working directory for the kernel
cd "$WORKING_DIR" || exit

# setup ipython extensions
cp -r /etc/ipython/ /home/notebooks/.ipython/

# clear out kubernetes_specific env vars before starting kernel gateway as it will inherit them
prefix="KUBERNETES_"; for var in $(printenv | cut -d= -f1); do [[ "$var" == "$prefix"* ]] && unset "$var"; done

exec jupyter kernelgateway --config=/etc/system/kernel/jupyter_kernel_gateway_config.py --debug
