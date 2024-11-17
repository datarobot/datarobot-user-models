#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pushd ${script_dir}/../custom_model_runner/
make dist
cp dist/datarobot_drum-1.14.2-py3-none-any.whl ${script_dir}/python3_sklearn/
popd

pushd $script_dir
tar cvzf python3_sklearn_s5cmd.tar.gz -C ${script_dir}/python3_sklearn/ .
popd
