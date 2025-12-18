#!/usr/bin/env bash

source tools/image-build-utils.sh
build_drum
DRUM_WHEEL_REAL_PATH="$(realpath "$(find ./custom_model_runner/dist/*.whl)")"
build_all_dropin_env_dockerfiles "${DRUM_WHEEL_REAL_PATH}"
