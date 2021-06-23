from gcr.io/deeplearning-platform-release/base-cu110

ENV CUDNN_VERSION 8.0.5.39

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=$CUDNN_VERSION-1+cuda11.0 \
    && apt-mark hold libcudnn8 && \
    rm -rf /var/lib/apt/lists/*


RUN conda install -c nvidia -c rapidsai -c numba -c conda-forge nvtabular python=3.7 cudatoolkit=11.0 -y

RUN pip install tensorflow==2.4.1
RUN pip install nvidia-pyindex
RUN pip install tritonclient

COPY src/ src/

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV TF_MEMORY_ALLOCATION=0.7