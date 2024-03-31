# This is the default base image for use with user models and workflows.
FROM nvcr.io/ohlfw0olaadg/ea-participants/nemollm-inference-ms:24.01
USER root
# Need to change the UID of nemo user until upstream fixes this as
# it is out-of-range for our rootless buildkit image builder.
RUN groupmod -g 1000 nemo && \
  usermod -u 1000 -g 1000 nemo && \
  chown -R 1000.1000 /home/nemo /model-store && \
  rm /var/log/lastlog

# Need to do this to squash our hack above so rootless buildkit won't
# try and unpack any layers with out-of-bound UIDs
FROM scratch
COPY --from=0 / /
ARG BASE_PATH=/opt/tritonserver/bin:/usr/local/cmake/bin:/usr/local/cmake/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/tritonserver/bin

ENV PATH=${BASE_PATH} \
    CUDA_VERSION=12.3.2.001 \
    CUDA_DRIVER_VERSION=545.23.08 \
    CUDA_CACHE_DISABLE=1 \
    NVIDIA_REQUIRE_JETPACK_HOST_MOUNTS= \
    _CUDA_COMPAT_PATH=/usr/local/cuda/compat \
    ENV=/etc/shinit_v2 \
    BASH_ENV=/etc/bash.bashrc \
    SHELL=/bin/bash \
    NVIDIA_REQUIRE_CUDA=cuda>=9.0 \
    NCCL_VERSION=2.19.3 \
    CUBLAS_VERSION=12.3.4.1 \
    CUFFT_VERSION=11.0.12.1 \
    CURAND_VERSION=10.3.4.107 \
    CUSPARSE_VERSION=12.2.0.103 \
    CUSOLVER_VERSION=11.5.4.101 \
    CUTENSOR_VERSION=1.7.0.1 \
    NPP_VERSION=12.2.3.2 \
    NVJPEG_VERSION=12.3.0.81 \
    CUDNN_VERSION=8.9.7.29+cuda12.2 \
    TRT_VERSION=8.6.1.6+cuda12.0.1.011 \
    TRTOSS_VERSION=23.11 \
    NSIGHT_SYSTEMS_VERSION=2023.4.1.97 \
    NSIGHT_COMPUTE_VERSION=2023.3.1.1 \
    DALI_VERSION=1.32.0 \
    DALI_BUILD=10610166 \
    POLYGRAPHY_VERSION=0.49.1 \
    TRANSFORMER_ENGINE_VERSION=1.1 \
    LD_LIBRARY_PATH=/usr/local/tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/opt/tritonserver/backends/tensorrtllm \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    NVIDIA_PRODUCT_NAME="Triton Server Base" \
    GDRCOPY_VERSION=2.3 \
    HPCX_VERSION=2.16rc4 \
    MOFED_VERSION=5.4-rdmacore39.0 \
    OPENUCX_VERSION=1.15.0 \
    OPENMPI_VERSION=4.1.5rc2 \
    RDMACORE_VERSION=39.0 \
    OPAL_PREFIX=/opt/hpcx/ompi \
    OMPI_MCA_coll_hcoll_enable=0 \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs: \
    NVIDIA_TRITON_SERVER_BASE_VERSION=23.12 \
    NVIDIA_BUILD_ID=77457706 \
    DEBIAN_FRONTEND=noninteractive \
    TRT_ROOT=/usr/local/tensorrt \
    PYTHONPATH=/opt/nemollm/FasterTransformer:/opt/nemollm/nemo-fw: \
    NIM_GIT_COMMIT=7a70adeb \
    NEMO_USER=nemo


USER root
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    zstd \
  && rm -rf /var/lib/apt/lists/*

USER 1000
ENV MODEL_STORE_PATH=/model-store
ENV DATAROBOT_VENV_PATH=/home/nemo/dr
ENV PIP_NO_CACHE_DIR=1
ENV PATH=${BASE_PATH}
RUN python -m venv $DATAROBOT_VENV_PATH && $DATAROBOT_VENV_PATH/bin/pip install -U pip
COPY dr_requirements.txt dr_requirements.txt

# artifact downloader tool
COPY artifact_downloader/ ./artifact_downloader/
ENV PYTHONPATH=$PYTHONPATH:${CODE_DIR}/artifact_downloader/

RUN $DATAROBOT_VENV_PATH/bin/pip install -r dr_requirements.txt

ENV CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080
WORKDIR ${CODE_DIR}
COPY --chown=1000:1000 ./*.sh ./*.py ${CODE_DIR}/


ENTRYPOINT ["/opt/code/start_server.sh"]