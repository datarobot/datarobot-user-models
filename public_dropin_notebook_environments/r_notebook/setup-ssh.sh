#!/bin/bash

echo "Persisting container environment variables for sshd..."
{
    echo "#!/bin/bash"
    echo "# This file is auto-populated with kernel env vars on container creation"
    echo "# to ensure that they are exposed in ssh sessions"
    echo "# Ref: https://github.com/jenkinsci/docker-ssh-agent/issues/33#issuecomment-597367846"
    echo "set -a"
    # set -a ensures that all modified/added shell variables are exported
    # ignore PWD/HOME/SHLVL/_ because these are specific to the current user and session
    # ignore TERM because it is set by asyncssh
    # ignore LD_PRELOAD for various security risks
    # ignore PS1 because it is set in setup-shell.sh
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
