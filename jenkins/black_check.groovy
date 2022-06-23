node('multi-executor && ubuntu:focal'){
  checkout scm
  sh """#!/bin/bash
  set -xe
  virtualenv .venv -p python3.8
  source .venv/bin/activate
  pip install -U pip
  pip install --only-binary=:all: -r requirements_lint.txt
  black --check .
  """
}