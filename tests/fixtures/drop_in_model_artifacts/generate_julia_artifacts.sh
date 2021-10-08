#!/bin/bash
cd /opt/drum
julia -J/opt/julia/sys.so --project=/opt/julia /opt/drum/tests/fixtures/drop_in_model_artifacts/generate_julia_artifacts.jl



# /opt/drum/tests/fixtures/drop_in_model_artifacts/generate_julia_artifacts.sh