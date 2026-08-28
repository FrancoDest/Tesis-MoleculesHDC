"""Capa compacta del pipeline HDC montado arriba de Mole-BERT.

Mantiene intacto el codigo del repo original y concentra solo la logica
agregada del flujo embeddings -> JL -> HDC -> training/eval.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from rdkit import Chem
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

ORIGINAL_DIR = Path(__file__).resolve().parents[1] / "original"
if str(ORIGINAL_DIR) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_DIR))

from loader import mol_to_graph_data_obj_simple
from model import GNN


DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 800
DEFAULT_ITERATIONS = 100

POOLING_FUNCTIONS = {
    "mean": global_mean_pool,
    "sum": global_add_pool,
    "max": global_max_pool,
}


def load_dataset_rows(
    input_csv: str,
    smiles_column: str,
    target_column: str,
    extra_columns: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(input_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"El archivo {input_csv} no tiene encabezados.")
        if smiles_column not in reader.fieldnames:
            raise ValueError(f"No encontre la columna '{smiles_column}' en {input_csv}.")
        if target_column not in reader.fieldnames:
            raise ValueError(f"No encontre la columna '{target_column}' en {input_csv}.")
        missing = [column for column in extra_columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"No encontre estas columnas en {input_csv}: {missing}")
        for index, row in enumerate(reader):
            smiles = (row.get(smiles_column) or "").strip()
            target = (row.get(target_column) or "").strip()
            if not smiles or target == "":
                continue
            parsed_row = {
                "id": row.get("id") or str(index),
                "smiles": smiles,
                "target": target,
            }
            for column in extra_columns:
                parsed_row[column] = row.get(column, "")
            rows.append(parsed_row)
    return rows


def build_molecular_graphs(dataset_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list]:
    valid_rows: list[dict[str, str]] = []
    molecular_graphs = []
    for row in dataset_rows:
        smiles = row["smiles"]
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        molecular_graphs.append(mol_to_graph_data_obj_simple(mol))
        valid_rows.append(row)
    if not molecular_graphs:
        raise ValueError("No se pudo convertir ningun SMILES valido a grafo.")
    return valid_rows, molecular_graphs


def build_encoder(
    checkpoint: str,
    device: torch.device,
    num_layer: int = 5,
    emb_dim: int = 300,
    jk: str = "last",
    dropout_ratio: float = 0.0,
    gnn_type: str = "gin",
) -> GNN:
    encoder = GNN(
        num_layer=num_layer,
        emb_dim=emb_dim,
        JK=jk,
        drop_ratio=dropout_ratio,
        gnn_type=gnn_type,
    )
    state_dict = torch.load(checkpoint, map_location=device)
    encoder.load_state_dict(state_dict)
    encoder.to(device)
    encoder.eval()
    return encoder


@torch.no_grad()
def compute_graph_embeddings(
    encoder: GNN,
    molecular_graphs: list,
    pooling: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    pooled_embeddings = []
    pool_fn = POOLING_FUNCTIONS[pooling]
    for start in range(0, len(molecular_graphs), batch_size):
        batch_graphs = molecular_graphs[start : start + batch_size]
        batch = Batch.from_data_list(batch_graphs).to(device)
        node_embeddings = encoder(batch.x, batch.edge_index, batch.edge_attr)
        graph_embeddings = pool_fn(node_embeddings, batch.batch)
        pooled_embeddings.append(graph_embeddings.cpu().numpy())
    return np.concatenate(pooled_embeddings, axis=0)


def generate_embeddings_from_rows(
    dataset_rows: list[dict[str, str]],
    checkpoint: str,
    pooling: str = "mean",
    batch_size: int = 64,
) -> tuple[list[dict[str, str]], np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_rows, molecular_graphs = build_molecular_graphs(dataset_rows)
    encoder = build_encoder(checkpoint=checkpoint, device=device)
    embeddings = compute_graph_embeddings(
        encoder=encoder,
        molecular_graphs=molecular_graphs,
        pooling=pooling,
        batch_size=batch_size,
        device=device,
    )
    return valid_rows, embeddings


def convert_hypervectors(matrix: np.ndarray) -> np.ndarray:
    return np.where(matrix >= 0.0, 1, -1).astype(np.int8)


def write_matrix_csv(
    output_path: str,
    rows: list[dict[str, str]],
    matrix: np.ndarray,
    prefix: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["id", "smiles"] + [f"{prefix}_{index:05d}" for index in range(matrix.shape[1])]
        writer.writerow(header)
        for row, vector in zip(rows, matrix):
            writer.writerow([row["id"], row["smiles"], *vector.tolist()])


def parse_binary_target(value: str) -> int:
    return int(float(str(value).strip()))


def score_for_predictions_csv(predicted_proba: np.ndarray, predicted_labels: np.ndarray) -> np.ndarray:
    """Probabilidad a guardar en las predicciones: para binario, la de la
    clase positiva (comportamiento historico); para multiclase, la de la clase predicha."""
    if predicted_proba.shape[1] == 2:
        return predicted_proba[:, 1]
    return predicted_proba[np.arange(len(predicted_labels)), predicted_labels]


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, object]:
    """y_proba: matriz completa de probabilidades por clase (n_samples,
    n_classes). Si el target tiene mas de 2 clases , se usa promedio macro y ROC AUC one-vs-rest."""
    is_multiclass = y_proba.shape[1] > 2
    average = "macro" if is_multiclass else "binary"
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).astype(int).tolist(),
    }
    try:
        if is_multiclass:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        else:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
    except ValueError:
        metrics["roc_auc"] = 0.5
    return metrics


def train_and_evaluate_classifier(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, dict[str, float]]:
    model = SGDClassifier(
        loss="log_loss",
        random_state=random_state,
        max_iter=1000,
        tol=1e-3,
        average=True,
    )
    training_start = time.time()
    model.fit(X_train, y_train)
    training_seconds = time.time() - training_start

    testing_start = time.time()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = evaluate_classification(y_test, y_pred, y_proba)
    testing_seconds = time.time() - testing_start
    return metrics, y_pred, y_proba, {
        "training_seconds": float(training_seconds),
        "testing_seconds": float(testing_seconds),
    }


def compute_average_confusion_matrix(confusion_matrices: list[list[list[int]]]) -> list[list[float]]:
    matrix = np.asarray(confusion_matrices, dtype=np.float64)
    return matrix.mean(axis=0).tolist()


def write_prediction_rows_csv(path: str, rows: list[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["iteration", "random_state", "id", "smiles", "y_true", "y_pred", "y_score"],
        )
        writer.writeheader()
        writer.writerows(rows)


def get_process_cpu_time_seconds(process: psutil.Process) -> float:
    cpu_times = process.cpu_times()
    return float(cpu_times.user + cpu_times.system)
