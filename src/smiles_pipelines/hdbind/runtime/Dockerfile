FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch CPU-only: HDBind solo necesita tensores/algebra lineal simple
# (proyeccion aleatoria + similitud coseno), no hace falta CUDA ni torch-hd
# (a diferencia de graphHD, aca no se usan tipos VSA, solo tensores planos
# con similitud coseno, que es invariante a la escala del vector).
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.1 \
    scikit-learn==1.4.1.post1 \
    psutil==5.9.8 \
    rdkit-pypi==2022.9.5

WORKDIR /opt/hdbind

COPY . /opt/hdbind

WORKDIR /workspace

ENTRYPOINT ["python", "/opt/hdbind/pipeline/run_hdbind.py"]
CMD []
