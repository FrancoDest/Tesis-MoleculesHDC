FROM mambaorg/micromamba:1.5.10

WORKDIR /workspace

RUN micromamba create -y -n logistic-regression -c conda-forge \
    python=3.11 \
    numpy \
    pandas \
    scikit-learn \
    rdkit \
    psutil && \
    micromamba clean --all --yes

SHELL ["micromamba", "run", "-n", "logistic-regression", "/bin/bash", "-lc"]

ENTRYPOINT ["micromamba", "run", "-n", "logistic-regression", "python", "train_logistic_regression.py"]
CMD []
