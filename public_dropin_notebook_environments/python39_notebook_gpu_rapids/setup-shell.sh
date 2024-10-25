#!/bin/bash

PROMPT_COMMAND='history -a; history -n'
PS1='[\[\033[38;5;172m\]\u\[$(tput sgr0)\]@kernel \[$(tput sgr0)\]\[\033[38;5;39m\]\w\[$(tput sgr0)\]]\$ \[$(tput sgr0)\]'

# custom bash history path if 'storage' directory exists
if [ -d "/home/notebooks/storage" ]; then
  HISTFILE='/home/notebooks/storage/.bash_history'
fi

# shellcheck disable=SC1091
source /etc/system/kernel/setup-venv.sh
