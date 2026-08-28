FROM mambaorg/micromamba:1.5.10

WORKDIR /workspace

RUN micromamba create -y -n dummy-classifier -c conda-forge \
    python=3.11 \
    numpy \
    pandas \
    scikit-learn \
    rdkit \
    psutil && \
    micromamba clean --all --yes

SHELL ["micromamba", "run", "-n", "dummy-classifier", "/bin/bash", "-lc"]

ENTRYPOINT ["micromamba", "run", "-n", "dummy-classifier", "python", "train_dummy_classifier.py"]
CMD []
