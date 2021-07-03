#!/bin/bash

. "/opt/conda/etc/profile.d/conda.sh"
conda activate nvtabular_dev_11.0

pip install -e NVTabular/.
pip install nvidia-pyindex
pip install tritonclient

echo "This is entrypoint!"
echo "Current conda environment:" $CONDA_DEFAULT_ENV

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TF_MEMORY_ALLOCATION=0.7

exec "$@"