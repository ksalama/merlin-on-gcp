from gcr.io/deeplearning-platform-release/base-cu110

WORKDIR /movielens

ENV LD_LIBRARY_PATH /usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

COPY NVTabular/ NVTabular/ 

RUN conda env create -f=NVTabular/conda/environments/nvtabular_dev_cuda11.0.yml 

COPY entrypoint.sh ./
COPY src/ src/

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
