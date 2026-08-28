FROM --platform=linux/amd64 python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.1 \
    scikit-learn==1.4.1.post1 \
    scipy==1.12.0 \
    rdkit-pypi==2022.9.5 \
    deepchem==2.8.0

WORKDIR /opt/nci_prep

COPY nci_dataset_prep_docker/prepare_nci_dataset.py /opt/nci_prep/prepare_nci_dataset.py

CMD ["/bin/bash"]
