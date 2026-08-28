FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# Mol2vec usa APIs viejas de gensim, asi que conviene mantener un stack conservador.
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.12.0 \
    pandas==2.2.1 \
    joblib==1.3.2 \
    psutil==5.9.8 \
    tqdm==4.66.2 \
    gensim==4.3.2 \
    scikit-learn==1.4.1.post1 \
    rdkit-pypi==2022.9.5

WORKDIR /opt/mol2vec

RUN mkdir -p /opt/mol2vec/examples/models
ADD https://raw.githubusercontent.com/samoturk/mol2vec/master/examples/models/model_300dim.pkl /opt/mol2vec/examples/models/model_300dim.pkl

COPY . /opt/mol2vec

COPY runtime/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
