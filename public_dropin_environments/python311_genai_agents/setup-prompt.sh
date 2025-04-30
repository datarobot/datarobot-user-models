#!/bin/bash

PS1='[\[\033[38;5;172m\]\u\[$(tput sgr0)\]@kernel \[$(tput sgr0)\]\[\033[38;5;39m\]\w\[$(tput sgr0)\]]\$ \[$(tput sgr0)\]'

# shellcheck disable=SC1091
source /etc/system/kernel/setup-venv.sh
