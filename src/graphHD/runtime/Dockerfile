FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch CPU-only para no arrastrar dependencias CUDA gigantes que este
# pipeline no usa.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2

# Nota sincera: el requirements original viene congelado desde otro entorno y
# no es reproducible tal cual. Este runtime instala solo lo necesario para el
# runner original adaptado a datasets moleculares tabulares.
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.1 \
    scikit-learn==1.4.1.post1 \
    psutil==5.9.8 \
    torch-hd==5.7.1 \
    rdkit-pypi==2022.9.5

WORKDIR /opt/graphHD

COPY . /opt/graphHD

COPY runtime/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
