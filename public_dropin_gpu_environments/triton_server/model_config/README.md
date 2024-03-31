This folder contains the Triton model configuration parser generated from the protocol buffer scheme (r24.02)
https://github.com/triton-inference-server/common/blob/r24.02/protobuf/model_config.proto
```shell
wget https://github.com/triton-inference-server/common/blob/r24.02/protobuf/model_config.proto
protoc --proto_path=src --python_out=model_config/triton model_config.proto
```
Please do not modify or delete this folder from your Docker context.
