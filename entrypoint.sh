#!/bin/bash

. "/opt/conda/etc/profile.d/conda.sh"
conda activate nvtabular_dev_11.0

pip install -e NVTabular/.
pip install nvidia-pyindex
pip install tritonclient
pip install tensorflow==2.4.1

echo "Current conda environment:" $CONDA_DEFAULT_ENV

exec "$@"
