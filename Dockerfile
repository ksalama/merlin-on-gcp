from nvcr.io/nvidia/nvtabular:0.3

RUN pip install tensorflow==2.4.1 gcsfs nvidia-pyindex 
RUN pip install tritonclient

COPY src/ src/

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV TF_MEMORY_ALLOCATION=0.7