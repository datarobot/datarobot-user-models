#/usr/bin/env bash

VIRTUAL_ENV="${VIRTUAL_ENV:-}"
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Deactivating existing virtual environment: $VIRTUAL_ENV"
    source "$VIRTUAL_ENV/bin/activate"
    deactivate
fi

tmp_venv_path=/tmp/venv
echo "Creating and sourcing a new virtual environment for the tests: $tmp_venv_path"
rm -rf $tmp_venv_path
python3 -m venv $tmp_venv_path
. ${tmp_venv_path}/bin/activate

pip install -U pip
pip install wheel
echo "Virtual environment is ready for the tests: $tmp_venv_path"