#!/usr/bin/env bash

script_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
. ${script_dir}/update-python-to-meet-requirements.sh

VIRTUAL_ENV="${VIRTUAL_ENV:-}"
if [ -n "$VIRTUAL_ENV" ]; then
    echo "== Deactivating existing virtual environment '$VIRTUAL_ENV' =="
    source "$VIRTUAL_ENV/bin/activate"
    deactivate
fi

tmp_venv_path=/tmp/venv
echo "== Creating and sourcing a new virtual environment for the tests: '$tmp_venv_path' =="
rm -rf $tmp_venv_path
python3 -m venv $tmp_venv_path
. ${tmp_venv_path}/bin/activate
pip install -U pip
# setuptools>=82 + wheel>=0.47 break datarobot-drum's wheel build with
# "[Errno 17] File exists: ...dist-info" (install_lib + install_egg_info
# both write dist-info). Pin until upstream fixes it.
pip install -U 'setuptools<82' 'wheel<0.47'
echo "== Virtual environment is ready for the tests: '$tmp_venv_path' =="
