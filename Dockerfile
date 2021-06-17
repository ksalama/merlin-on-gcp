from gcr.io/deeplearning-platform-release/base-cu110

ENV CUDNN_VERSION 8.0.5.39

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=$CUDNN_VERSION-1+cuda11.0 \
    && apt-mark hold libcudnn8 && \
    rm -rf /var/lib/apt/lists/*


COPY nvt-cuda11.0.yaml nvt-cuda11.0.yaml

RUN conda env create -f=nvt-cuda11.0.yaml

ENV PATH /opt/conda/envs/nvt-cuda11.0/bin:$PATH

RUN pip install tritonclient

COPY src/ src/

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python