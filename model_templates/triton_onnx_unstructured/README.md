<!--
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Triton ONNX model template

This document describes how to deploy a simple ResNet model with DataRobot custom models into
Triton Inference Server environment.

## Step 1: Feature flags (Public Preview feature)

To use the Triton Inference Server drop-in environment, turn on the feature flags:
- `ENABLE_CUSTOM_MODEL_GPU_INFERENCE`
- `ENABLE_MLOPS_RESOURCE_REQUEST_BUNDLES`

## Step 2: Set up a new Custom Model version 

- Create a new Custom Model with the Target Type `unstructured`. 
- Select the `[NVIDIA] Triton Inference Server` Environment.
- Upload the whole `model_repository` directory into the custom model. The expected model structure is:
```
model_repository
|
+-- densenet_onnx
    |
    +-- config.pbtxt
    +-- 1
        |
        +-- model.onnx
```

## Step 3: Set up the Resource Bundle for a Custom Model version
 
The `densenet_onnx` model runs on GPU and CPU-only instances. The minimal Resource Bundle configuration is:
- for CPU instance is M (1CPU, 1GB RAM)
- for GPU instance is GPU-S

Now, model is ready for deployment. Press `Register model` and then `Deploy`


## Step 4: Using a DataRobot Client to run predictions

Install dependencies & download an example image to test inference.
```
cd model_templates/triton_onnx_unstructured/client
pip install -r requirements.txt

wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
```

Now we need to modify the client's code and set the following keys:
```
$ vim model_templates/triton_onnx_unstructured/client/datarobot-predict.py

API_URL
API_KEY
DATAROBOT_KEY
DEPLOYMENT_ID
```

To find correct values, go to the DataRobot Console and choose your deployment. Then switch to the 
Predictions / Predictions API / Prediction Type Real-time / Show secrets.   

Run the predictions:
```
python model_templates/triton_onnx_unstructured/client/datarobot-predict.py
```
