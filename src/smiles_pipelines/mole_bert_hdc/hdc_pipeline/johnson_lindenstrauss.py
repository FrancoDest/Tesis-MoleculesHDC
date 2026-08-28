"""Helpers de proyeccion Johnson-Lindenstrauss del pipeline agregado."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.random_projection import SparseRandomProjection


def fit_projection_model(
    embeddings: np.ndarray,
    jl_dim: int,
    random_state: int,
) -> SparseRandomProjection:
    projection_model = SparseRandomProjection(
        n_components=jl_dim,
        dense_output=True,
        random_state=random_state,
    )
    projection_model.fit(csr_matrix(embeddings))
    return projection_model


def apply_projection_model(
    projection_model: SparseRandomProjection,
    embeddings: np.ndarray,
) -> np.ndarray:
    return projection_model.transform(csr_matrix(embeddings))
