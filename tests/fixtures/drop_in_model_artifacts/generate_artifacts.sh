set -e
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# 
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
rm -rf SKLearn.py Keras.py XGBoost.py
python3 -m jupyter nbconvert --execute --to python  SKLearn.ipynb
python3 SKLearn.py || rm -rf SKLearn.py

python3 -m jupyter nbconvert --execute --to python  Keras.ipynb
python3 Keras.py || rm -rf Keras.py

python3 -m jupyter nbconvert --execute --to python  XGBoost.ipynb
python3 XGBoost.py || rm -rf XGBoost.py

python3 PyTorch.py

# R models.
# For the regression model has to rename the target column in Juniors dataset "Grade 2014" -> "Grade_2014"
Rscript Rmodel.R

## create the h2o model artifacts
## might require a pip install of h2o==3.34.0.1
python3 generate_h2o_artifacts.py

## create julia model artifacts
## move the root of drum 
cd ../../..
## change permissions on generate_julia_artifacts.sh 
chmod 777 tests/fixtures/drop_in_model_artifacts/generate_julia_artifacts.sh
## build docker image to run julia models.  not necessary if you want to install julia 1.5.4 and all dependencies
## might take 10 minutes or so to build. 
docker build --tag julia_env docker/julia_dropin_env_base
docker run -i -v $(pwd):/opt/drum julia_env /opt/drum/tests/fixtures/drop_in_model_artifacts/generate_julia_artifacts.sh
