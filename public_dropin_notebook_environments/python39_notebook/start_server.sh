#!/bin/bash
cd /nbx/agent
source /nbx/agent/.venv/bin/activate
nohup uvicorn agent:app --host 0.0.0.0 --port 8889 &

cp -L /var/run/nbx/ssh/authorized_keys/nbx /etc/authorized_keys/ && chmod 600 /etc/authorized_keys/nbx
mkdir /etc/ssh/keys && cp -L /var/run/nbx/ssh/keys/ssh_host_* /etc/ssh/keys/ && chmod 600 /etc/ssh/keys/ssh_host_*
nohup /usr/sbin/sshd -D &

cd /nbx
source /nbx/.venv/bin/activate
jupyter kernelgateway --config=./jupyter_kernel_gateway_config.py --debug