#!/usr/bin/env bash

set -exuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${script_dir}/../../tools/create-and-source-venv.sh

pip install -r requirements_test_unit.txt
# Install datarobot-drum from a per-step copy in container-local /tmp: the
# checkout is shared across the parallel python-version matrix steps, and
# concurrent wheel builds collide in the shared custom_model_runner/build/
# directory ("[Errno 17] File exists: ...dist-info").
cmr_dir="$(mktemp -d)/custom_model_runner"
cp -a custom_model_runner "$cmr_dir"
pip install "$cmr_dir"
pytest -v tests/unit --junit-xml=results.tests.xml