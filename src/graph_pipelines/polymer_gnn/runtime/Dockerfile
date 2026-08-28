FROM --platform=linux/amd64 python:3.10-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.12.0 \
    pandas==2.2.1 \
    scikit-learn==1.4.1.post1 \
    psutil==5.9.8 \
    rdkit-pypi==2022.9.5 \
    tqdm==4.66.2

RUN pip install --no-cache-dir \
    pyg_lib \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.2+cpu.html

RUN pip install --no-cache-dir torch-geometric==2.5.3

WORKDIR /workspace

CMD ["/bin/bash"]
