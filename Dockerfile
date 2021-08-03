from gcr.io/deeplearning-platform-release/base-cu110

WORKDIR /movielens

ENV LD_LIBRARY_PATH /usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

COPY NVTabular/ NVTabular/

COPY nvtabular_dev_cuda11.0.yaml ./
RUN conda env create -f=nvtabular_dev_cuda11.0.yaml

COPY entrypoint.sh ./

COPY src/ src/

RUN chmod +x entrypoint.sh

RUN /opt/conda/envs/nvtabular_dev_11.0/bin/pip install tritonclient

RUN /opt/conda/envs/nvtabular_dev_11.0/bin/pip install -e NVTabular/.

ENTRYPOINT ["./entrypoint.sh"]
