#/usr/bin/env bash

min_required_version="3.9"  # Required for datarobot-mlops
current_version=$(python3 --version 2>&1 | awk '{print $2}')

# Compare versions
if [[ $(echo -e "$current_version\n$min_required_version" | sort -V | head -n 1) != "$min_required_version" ]]; then
    echo "== Python version is less than $min_required_version. Updating... =="
    set -exuo pipefail
    apt-get update
    apt-get install -y \
      python${min_required_version} \
      python${min_required_version}-venv \
      python${min_required_version}-doc \
      binfmt-support
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${min_required_version} 1
    update-alternatives --set python3 /usr/bin/python${min_required_version}
    python3 --version
    apt-get install -y python3-pip
    python3 -m pip --version
    python3 -m pip install --upgrade setuptools
    python3 -m pip install --upgrade wheel
else
    echo "Python version is $current_version, which meets the requirement."
fi
