#/usr/bin/env bash

VIRTUAL_ENV="${VIRTUAL_ENV:-}"
if [ -n "$VIRTUAL_ENV" ]; then
    source "$VIRTUAL_ENV/bin/activate"
    deactivate
fi

tmp_venv_path=/tmp/venv
rm -rf $tmp_venv_path
python3 -m venv $tmp_venv_path
. ${tmp_venv_path}/bin/activate

pip install -U pip
pip install wheel
