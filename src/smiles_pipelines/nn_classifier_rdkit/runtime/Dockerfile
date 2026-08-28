FROM mambaorg/micromamba:1.5.10

WORKDIR /workspace

RUN micromamba create -y -n nn-classifier -c conda-forge \
    python=3.11 \
    numpy \
    pandas \
    scikit-learn \
    rdkit \
    psutil && \
    micromamba clean --all --yes

SHELL ["micromamba", "run", "-n", "nn-classifier", "/bin/bash", "-lc"]

ENTRYPOINT ["micromamba", "run", "-n", "nn-classifier", "python", "train_nn_classifier.py"]
CMD []
