FROM python:3.8-slim-bullseye

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# Stack conservador alineado con la antiguedad de MoleHD.
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    pandas==1.3.1 \
    scipy==1.10.1 \
    scikit-learn==1.2.2 \
    imbalanced-learn==0.8.0 \
    tqdm==4.62.1 \
    SmilesPE==0.0.3 \
    psutil==5.9.8 \
    rdkit-pypi==2022.9.5

WORKDIR /opt/molehd

COPY . /opt/molehd

COPY runtime/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
