#!/bin/bash

echo "Persisting container environment variables for sshd..."
{
    echo "#!/bin/bash"
    echo "# This file is auto-populated with kernel env vars on container creation"
    echo "# to ensure that they are exposed in ssh sessions"
    echo "# Ref: https://github.com/jenkinsci/docker-ssh-agent/issues/33#issuecomment-597367846"
    echo "set -a"
    env | grep -E -v "^(PWD=|HOME=|TERM=|SHLVL=|LD_PRELOAD=|PS1=|_=|KUBERNETES_)" | while read -r line; do
      NAME=$(echo "$line" | cut -d'=' -f1)
      VALUE=$(echo "$line" | cut -d'=' -f2-)
      # Use eval to handle complex cases like export commands with spaces
      echo "$NAME='$VALUE'"
    done
    echo "set +a"
    # setup the working directory for terminal sessions
    echo "cd $WORKING_DIR"
} > /etc/profile.d/notebooks-load-env.sh
