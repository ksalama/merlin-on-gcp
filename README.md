# NVIDIA Merlin on GCP
An example of running NVidia Merlin on Google Cloud

## Getting started

1. [Setting up MLOps environment](provision) on Google Cloud.
2. Start your AI Notebook instance (with CUDA Toolkit 11.0 installed V100 GPU(s)).
3. Open the JupyterLab then open a new Terminal
4. Create a virtual environment with NVTabular
    ```
    git clone https://github.com/NVIDIA/NVTabular.git
    cd NVTabular/
    conda env create -f=conda/environments/nvtabular_dev_cuda11.0.yml 
    conda activate nvtabular_dev_11.0
    python -m ipykernel install --user --name=nvt-11-0
    pip install -e .
    pip install gcsfs
    pip install tensorflow==2.4.1
    pip install nvidia-pyindex
    pip install tritonclient
    cd ~
    ```
5. Clone the repository to your AI Notebook instance:
    ```
    git clone https://github.com/ksalama/merlin-on-gcp.git
    cd merlin-on-gcp
    ```
6. Build the Docker container image used in Vertex AI
    ```
    export PROJECT=[your project name]
    export IMAGE_NAME=nvt-cuda11.0-tf2.4
    export IMAGE_URI=gcr.io/$PROJECT/$IMAGE_NAME
    echo $IMAGE_URI
    gcloud builds submit --tag $IMAGE_URI . --timeout=30m --machine-type=e2-highcpu-8
    ```


