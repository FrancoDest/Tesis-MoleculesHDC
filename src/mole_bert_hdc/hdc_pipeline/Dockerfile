FROM --platform=linux/amd64 python:3.10-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# Stack moderna y consistente para correr Mole-BERT en CPU.
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Dependencias generales del pipeline:
# - scikit-learn: proyeccion JL y evaluacion
# - rdkit: parseo de SMILES
# - pandas/scipy/numpy: manejo de datos
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.12.0 \
    pandas==2.2.1 \
    scikit-learn==1.4.1.post1 \
    networkx==3.2.1 \
    psutil==5.9.8 \
    rdkit-pypi==2022.9.5 \
    tqdm==4.66.2 \
    protobuf==3.20.3

# PyTorch Geometric y extensiones compiladas compatibles con torch 2.2 CPU.
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

# Uso recomendado:
# docker build --platform linux/amd64 -t molebert -f Mole-BERT/hdc_pipeline/Dockerfile Mole-BERT/hdc_pipeline
# docker run --rm -it --platform linux/amd64 -v "$(pwd):/tesis" -w /tesis/Mole-BERT molebert /bin/bash
#
# Ejemplo dentro del contenedor:
# python hdc_pipeline/run_hdc_pipeline.py --input-csv dataset/bbbp/raw/BBBP_rdkit_valid.csv --target-column p_np
