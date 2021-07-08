from gcr.io/deeplearning-platform-release/base-cu110

WORKDIR /movielens

RUN git clone https://github.com/NVIDIA/NVTabular.git

RUN conda env create -f=NVTabular/conda/environments/nvtabular_dev_cuda11.0.yml 
RUN conda install -n nvtabular_dev_11.0 tensorflow==2.4.1 gcsfs

COPY entrypoint.sh ./
COPY src/ src/

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
