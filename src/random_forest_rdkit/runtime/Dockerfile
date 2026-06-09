FROM mambaorg/micromamba:1.5.10

WORKDIR /workspace

RUN micromamba create -y -n bbbp-rf -c conda-forge \
    python=3.11 \
    numpy \
    pandas \
    scikit-learn \
    rdkit \
    psutil && \
    micromamba clean --all --yes

SHELL ["micromamba", "run", "-n", "bbbp-rf", "/bin/bash", "-lc"]

ENTRYPOINT ["micromamba", "run", "-n", "bbbp-rf", "python", "train_random_forest_bbbp.py"]
CMD []
