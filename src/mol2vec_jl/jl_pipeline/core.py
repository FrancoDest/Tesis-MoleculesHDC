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


def compute_classification_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    predicted_scores: np.ndarray,
) -> dict[str, object]:
    return {
        "Accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "AUROC": float(roc_auc_score(true_labels, predicted_scores)),
        "Bacc": float(balanced_accuracy_score(true_labels, predicted_labels)),
        "F1": float(f1_score(true_labels, predicted_labels, zero_division=0)),
        "Precision": float(precision_score(true_labels, predicted_labels, zero_division=0)),
        "Recall": float(recall_score(true_labels, predicted_labels, zero_division=0)),
        "ConfusionMatrix": confusion_matrix(true_labels, predicted_labels).astype(int).tolist(),
    }


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
    )
    training_start = time.time()
    classifier.fit(train_features, train_labels)
    training_seconds = time.time() - training_start

    testing_start = time.time()
    predicted_labels = classifier.predict(test_features)
    predicted_scores = classifier.decision_function(test_features)
    metrics = compute_classification_metrics(test_labels, predicted_labels, predicted_scores)
    testing_seconds = time.time() - testing_start
    return metrics, predicted_labels, predicted_scores, {
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
