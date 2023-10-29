FROM rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1

COPY requirements.txt .
RUN pip install -U pip wheel
RUN pip install -r requirements.txt

RUN git clone https://github.com/sremes/bitsandbytes-rocm.git && \
    cd bitsandbytes-rocm && \
    ROCM_HOME=/opt/rocm ROCM_TARGET=gfx1101 make hip && \
    pip install -e .

WORKDIR /app