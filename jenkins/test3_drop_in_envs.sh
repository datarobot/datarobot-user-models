#!/usr/bin/env bash

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"

# need to install this first to workaround quantum requirements
# and missing argparse-formatter wheel in DR artifactory
pip install pytest-runner
# installing DRUM into the test env is required for push test
pip install datarobot-drum
# requirements_test may install newer packages for testing, e.g. `datarobot`
pip install -r requirements_test.txt

py.test tests/functional/ --junit-xml="$CDIR/results_drop_in.xml"
