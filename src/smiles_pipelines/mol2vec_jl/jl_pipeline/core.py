"""Helpers compactos para embeddings, proyeccion JL y entrenamiento final."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
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


def convert_hypervectors(matrix: np.ndarray) -> np.ndarray:
    """Binariza la proyeccion JL a hipervectores bipolares {-1, +1}."""
    return np.where(matrix >= 0.0, 1, -1).astype(np.int8)


def score_for_predictions_csv(predicted_proba: np.ndarray, predicted_labels: np.ndarray) -> np.ndarray:
    """Probabilidad a guardar en las predicciones: para binario, la de la
    clase positiva (comportamiento historico); para multiclase, la de la clase predicha."""
    if predicted_proba.shape[1] == 2:
        return predicted_proba[:, 1]
    return predicted_proba[np.arange(len(predicted_labels)), predicted_labels]


def compute_classification_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    predicted_proba: np.ndarray,
) -> dict[str, object]:
    """predicted_proba: matriz completa de probabilidades por clase
    (n_samples, n_classes). Si el target tiene mas de 2 clases , se usa promedio macro y ROC AUC one-vs-rest en vez
    del esquema binario simple."""
    is_multiclass = predicted_proba.shape[1] > 2
    average = "macro" if is_multiclass else "binary"
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted_labels)),
        "f1": float(f1_score(true_labels, predicted_labels, average=average, zero_division=0)),
        "precision": float(precision_score(true_labels, predicted_labels, average=average, zero_division=0)),
        "recall": float(recall_score(true_labels, predicted_labels, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels).astype(int).tolist(),
    }
    try:
        if is_multiclass:
            metrics["roc_auc"] = float(
                roc_auc_score(true_labels, predicted_proba, multi_class="ovr", average="macro")
            )
        else:
            metrics["roc_auc"] = float(roc_auc_score(true_labels, predicted_proba[:, 1]))
    except ValueError:
        metrics["roc_auc"] = 0.5
    return metrics


def train_and_evaluate_projection_classifier(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    random_state: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, dict[str, float]]:
    classifier = SGDClassifier(
        loss="log_loss",
        random_state=random_state,
        max_iter=1000,
        tol=1e-3,
        average=True,
        class_weight="balanced",
    )
    training_start = time.time()
    classifier.fit(train_features, train_labels)
    training_seconds = time.time() - training_start

    testing_start = time.time()
    predicted_labels = classifier.predict(test_features)
    predicted_proba = classifier.predict_proba(test_features)
    metrics = compute_classification_metrics(test_labels, predicted_labels, predicted_proba)
    testing_seconds = time.time() - testing_start
    return metrics, predicted_labels, predicted_proba, {
        "training_seconds": float(training_seconds),
        "testing_seconds": float(testing_seconds),
    }


def compute_average_confusion_matrix(confusion_matrices: list[list[list[int]]]) -> list[list[float]]:
    confusion_array = np.asarray(confusion_matrices, dtype=np.float64)
    return confusion_array.mean(axis=0).tolist()


def write_prediction_rows_csv(output_path: str, prediction_rows: list[dict[str, object]]) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["iteration", "random_state", "id", "smiles", "y_true", "y_pred", "y_score"],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)


def write_metrics_summary_json(output_path: str, payload: dict[str, object]) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
