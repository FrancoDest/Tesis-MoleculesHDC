"""Helpers de proyeccion Johnson-Lindenstrauss del pipeline agregado."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.random_projection import SparseRandomProjection


def load_embedding_table(path: str, id_column: str = "ID", smiles_column: str = "Smiles", feature_prefix: str = "mol2vec-") -> tuple[list[dict[str, str]], np.ndarray]:
    molecule_rows: list[dict[str, str]] = []
    feature_rows: list[list[float]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"El archivo {path} no tiene encabezados.")
        feature_columns = [column for column in reader.fieldnames if column.startswith(feature_prefix)]
        if not feature_columns:
            raise ValueError(f"No encontre columnas '{feature_prefix}*' en {path}.")
        for row in reader:
            row_id = (row.get(id_column) or "").strip()
            smiles = (row.get(smiles_column) or "").strip()
            if not row_id or not smiles:
                continue
            molecule_rows.append({"id": row_id, "smiles": smiles})
            feature_rows.append([float(row[column]) for column in feature_columns])
    if not feature_rows:
        raise ValueError(f"El archivo {path} no contiene embeddings validos.")
    return molecule_rows, np.asarray(feature_rows, dtype=np.float32)


def fit_projection_model(embeddings: np.ndarray, jl_dim: int, random_state: int) -> SparseRandomProjection:
    projection_model = SparseRandomProjection(
        n_components=jl_dim,
        dense_output=True,
        random_state=random_state,
    )
    projection_model.fit(csr_matrix(embeddings))
    return projection_model


def apply_projection_model(projection_model: SparseRandomProjection, embeddings: np.ndarray) -> np.ndarray:
    return np.asarray(projection_model.transform(csr_matrix(embeddings)), dtype=np.float32)


def write_projected_features_csv(output_path: str, molecule_rows: list[dict[str, str]], projected_matrix: np.ndarray, prefix: str = "jl") -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["id", "smiles"] + [f"{prefix}_{index:05d}" for index in range(projected_matrix.shape[1])]
        writer.writerow(header)
        for row, vector in zip(molecule_rows, projected_matrix):
            writer.writerow([row["id"], row["smiles"], *vector.tolist()])
