#!/bin/bash
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
cd /opt/drum
julia -J/opt/julia/sys.so --project=/opt/julia /opt/drum/tests/fixtures/drop_in_model_artifacts/generate_julia_artifacts.jl



# /opt/drum/tests/fixtures/drop_in_model_artifacts/generate_julia_artifacts.sh