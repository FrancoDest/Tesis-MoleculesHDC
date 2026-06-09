from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
import torchhd
from rdkit import Chem, RDLogger
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


DEFAULT_DIM = 10048
DEFAULT_ITERATIONS = 10
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 800
DEFAULT_PAGERANK_ALPHA = 0.85
DEFAULT_PAGERANK_ITERS = 100

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class MoleculeGraph:
    row_id: str
    smiles: str
    label: int
    edge_index: torch.Tensor
    num_nodes: int


def get_process_cpu_time_seconds(process: "psutil.Process | None") -> float | None:
    if process is None:
        return None
    cpu_times = process.cpu_times()
    return float(cpu_times.user + cpu_times.system)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta una variante de GraphHD mas fiel al notebook original, "
            "manteniendo entrada tabular desde SMILES."
        )
    )
    parser.add_argument("--dataset-csv", required=True, help="CSV con columnas de smiles y target.")
    parser.add_argument("--smiles-column", default="smiles", help="Nombre de la columna de SMILES.")
    parser.add_argument("--target-column", default="p_np", help="Nombre de la columna target.")
    parser.add_argument("--id-column", default="num", help="Nombre de la columna id original.")
    parser.add_argument("--output-dir", default="/run_outputs/artifacts", help="Directorio de salida.")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM, help="Dimension de los hipervectores.")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Cantidad de splits 80/20.")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Fraccion de test.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Semilla base.")
    parser.add_argument("--pagerank-alpha", type=float, default=DEFAULT_PAGERANK_ALPHA, help="Alpha de PageRank.")
    parser.add_argument("--pagerank-iters", type=int, default=DEFAULT_PAGERANK_ITERS, help="Iteraciones de PageRank.")
    return parser.parse_args()


def load_binary_graphs(
    dataset_csv: str,
    smiles_column: str,
    target_column: str,
    id_column: str,
) -> tuple[list[MoleculeGraph], int]:
    dataframe = pd.read_csv(dataset_csv)
    required = {smiles_column, target_column, id_column}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {dataset_csv}: {missing}")

    graphs: list[MoleculeGraph] = []
    invalid_rows = 0

    for _, row in dataframe.iterrows():
        smiles = str(row[smiles_column]).strip()
        target = row[target_column]
        row_id = str(row[id_column]).strip()
        if not smiles or pd.isna(target):
            invalid_rows += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_rows += 1
            continue

        edge_index = mol_to_edge_index(mol)
        graphs.append(
            MoleculeGraph(
                row_id=row_id,
                smiles=smiles,
                label=int(target),
                edge_index=edge_index,
                num_nodes=mol.GetNumAtoms(),
            )
        )

    if not graphs:
        raise ValueError("No quedaron moleculas validas para GraphHD.")
    return graphs, invalid_rows


def mol_to_edge_index(mol: Chem.Mol) -> torch.Tensor:
    edges: list[list[int]] = []
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        edges.append([begin_idx, end_idx])
        edges.append([end_idx, begin_idx])
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def sparse_stochastic_graph(graph: MoleculeGraph) -> torch.Tensor:
    rows, columns = graph.edge_index
    values_per_column = 1.0 / torch.bincount(columns, minlength=graph.num_nodes)
    values_per_edge = values_per_column[columns]
    size = (graph.num_nodes, graph.num_nodes)
    return torch.sparse_coo_tensor(graph.edge_index, values_per_edge, size)


def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.size(0), device=perm.device)
    return inverse


def to_undirected_unique(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index
    edge_index = edge_index.sort(dim=0)[0]
    return torch.unique(edge_index, dim=1)


def pagerank(graph: MoleculeGraph, alpha: float, max_iter: int) -> torch.Tensor:
    if graph.num_nodes == 0:
        return torch.empty((0,), dtype=torch.float32)
    if graph.num_nodes == 1:
        return torch.ones((1,), dtype=torch.float32)
    num_nodes = graph.num_nodes
    transition = sparse_stochastic_graph(graph).to(torch.float32) * alpha
    rank = torch.full((num_nodes,), 1.0 / num_nodes, dtype=torch.float32)
    teleport = torch.full((num_nodes,), (1.0 - alpha) / num_nodes, dtype=torch.float32)
    for _ in range(max_iter):
        rank = transition @ rank + teleport
    return rank


class GraphHDEncoder:
    def __init__(self, num_vectors: int, vector_size: int):
        self.node_ids = torch.randint(0, 2, (num_vectors, vector_size), dtype=torch.int32)
        self.vector_size = vector_size

    def encode(self, graph: MoleculeGraph, alpha: float, max_iter: int) -> torch.Tensor:
        if graph.num_nodes == 0:
            return torch.zeros(self.vector_size, dtype=torch.float32)

        pr = pagerank(graph, alpha=alpha, max_iter=max_iter)
        pr_argsort = inverse_permutation(torch.argsort(pr))
        node_id_hvs = self.node_ids[pr_argsort]
        undirected = to_undirected_unique(graph.edge_index)
        if undirected.numel() == 0:
            return torch.zeros(self.vector_size, dtype=torch.float32)

        row, col = undirected
        edge_hvs = [torch.bitwise_xor(node_id_hvs[s], node_id_hvs[t]) for s, t in zip(row, col)]
        graph_hv = reduce(torchhd.bundle, edge_hvs)
        return graph_hv.to(torch.float32)


class MajorityClassification:
    def __init__(self) -> None:
        self.class_vectors: Dict[int, torch.Tensor] = {}
        self.classes: list[int] = []
        self.class_to_index: Dict[int, int] = {}

    def add(self, sample: torch.Tensor, label: int) -> None:
        if label not in self.class_vectors:
            self.class_vectors[label] = sample
            self.classes.append(label)
            self.class_to_index = {class_label: idx for idx, class_label in enumerate(self.classes)}
            return
        self.class_vectors[label] = torchhd.bundle(self.class_vectors[label], sample)

    def score(self, sample: torch.Tensor) -> torch.Tensor:
        class_vectors = torch.stack([self.class_vectors[label] for label in self.classes])
        return torchhd.hamming_similarity(sample, class_vectors)

    def predict(self, sample: torch.Tensor) -> int:
        similarities = self.score(sample)
        max_index = int(torch.argmax(similarities).item())
        return self.classes[max_index]


def average_confusion_matrix(confusion_matrices: Sequence[Sequence[Sequence[int]]]) -> list[list[float]]:
    stacked = np.asarray(confusion_matrices, dtype=np.float64)
    return stacked.mean(axis=0).tolist()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No hay filas para escribir en {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_confusion_matrix(value: str | list[list[int]] | list[list[float]]) -> list[list[float]]:
    if isinstance(value, str):
        return json.loads(value)
    return value


def print_stats_block(title: str, row: dict[str, object]) -> None:
    print(title)
    print("Accuracy: ", row["Accuracy"])
    print("Auroc: ", row["AUROC"])
    print("Bacc: ", row["Balanced_Accuracy"])
    print("F1: ", row["F1_Score"])
    print("Precision: ", row["Precision"])
    print("Recall: ", row["Recall"])
    print("Confusion Matrix: ", parse_confusion_matrix(row["Confusion_Matrix"]))
    print("Random State: ", row["Random_State"])
    print("CPU Total Time: {:.2f}s".format(float(row["CPU_Total_Time_Seconds"] or 0.0)))
    print(
        "Memory Usage: {:.2f}% ({:.2f} MB)".format(
            float(row["Memory_Usage_%"] or 0.0),
            float(row["Memory_MB"] or 0.0),
        )
    )
    print()


def make_iteration_row(
    *,
    iteration: int,
    random_state: int,
    train_size: int,
    test_size: int,
    accuracy: float,
    auroc: float,
    balanced_accuracy: float,
    f1: float,
    precision: float,
    recall: float,
    confusion: list[list[int]],
    cpu_time_seconds: float | None,
    memory_percent: float | None,
    memory_mb: float | None,
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "Iteration": iteration,
        "Random_State": random_state,
        "Train_Size": train_size,
        "Test_Size": test_size,
        "Accuracy": float(accuracy),
        "AUROC": float(auroc),
        "Balanced_Accuracy": float(balanced_accuracy),
        "F1_Score": float(f1),
        "Precision": float(precision),
        "Recall": float(recall),
        "Confusion_Matrix": json.dumps(confusion, ensure_ascii=False),
        "CPU_Total_Time_Seconds": cpu_time_seconds,
        "Memory_Usage_%": memory_percent,
        "Memory_MB": memory_mb,
        "Elapsed_Seconds": float(elapsed_seconds),
        "Training_Seconds": None,
        "Testing_Seconds": None,
    }


def evaluate_iteration(
    encoder: GraphHDEncoder,
    train_graphs: Sequence[MoleculeGraph],
    test_graphs: Sequence[MoleculeGraph],
    classes: Sequence[int],
    pagerank_alpha: float,
    pagerank_iters: int,
) -> tuple[dict[str, float | list[list[int]]], list[dict[str, object]], float, float]:
    training_start = time.time()
    classifier = MajorityClassification()
    for graph in train_graphs:
        classifier.add(
            encoder.encode(graph, alpha=pagerank_alpha, max_iter=pagerank_iters),
            graph.label,
        )
    training_seconds = time.time() - training_start

    y_true: list[int] = []
    y_pred: list[int] = []
    positive_scores: list[float] = []
    prediction_rows: list[dict[str, object]] = []
    positive_class = max(classes)
    positive_index = classifier.class_to_index[positive_class]
    testing_start = time.time()

    for graph in test_graphs:
        hv = encoder.encode(graph, alpha=pagerank_alpha, max_iter=pagerank_iters)
        scores = classifier.score(hv)
        prediction = classifier.predict(hv)
        positive_score = float(scores[positive_index].item())
        y_true.append(graph.label)
        y_pred.append(prediction)
        positive_scores.append(positive_score)
        prediction_rows.append(
            {
                "id": graph.row_id,
                "smiles": graph.smiles,
                "y_true": graph.label,
                "y_pred": prediction,
                "score_positive_class": positive_score,
            }
        )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, positive_scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(classes)).tolist(),
    }
    testing_seconds = time.time() - testing_start
    return metrics, prediction_rows, training_seconds, testing_seconds


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.random_state)
    graphs, invalid_rows = load_binary_graphs(
        dataset_csv=args.dataset_csv,
        smiles_column=args.smiles_column,
        target_column=args.target_column,
        id_column=args.id_column,
    )
    labels = np.asarray([graph.label for graph in graphs], dtype=np.int64)
    classes = sorted({int(label) for label in labels.tolist()})
    class_counts = {label: int((labels == label).sum()) for label in classes}
    max_graph_size = max(graph.num_nodes for graph in graphs)
    process = psutil.Process() if psutil is not None else None
    encoder = GraphHDEncoder(num_vectors=max_graph_size, vector_size=args.dim)

    metrics_rows: list[dict[str, object]] = []
    best_iteration_predictions: list[dict[str, object]] | None = None
    best_iteration_record: dict[str, object] | None = None
    worst_iteration_record: dict[str, object] | None = None
    best_auroc = float("-inf")
    worst_auroc = float("inf")
    confusion_matrices: list[list[list[int]]] = []
    training_seconds_list: list[float] = []
    testing_seconds_list: list[float] = []

    dataset_stem = Path(args.dataset_csv).stem.lower()
    print("Usando variante graphHD mas fiel al notebook original...")
    print(f"Clase 0: {class_counts.get(0, 0)}, Clase 1: {class_counts.get(1, 0)}")
    print(f"Filas invalidas descartadas: {invalid_rows}")
    print(f"Iteraciones: {args.iterations}")
    print(f"Split por iteración: 80/20 con seed {args.random_state} + iteración")

    overall_start = time.time()
    for iteration in range(args.iterations):
        iteration_seed = args.random_state + iteration
        train_graphs, test_graphs = train_test_split(
            graphs,
            test_size=args.test_size,
            random_state=iteration_seed,
            stratify=labels,
        )
        start_time = time.time()
        cpu_time_start = get_process_cpu_time_seconds(process)

        metrics, predictions, training_seconds, testing_seconds = evaluate_iteration(
            encoder=encoder,
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            classes=classes,
            pagerank_alpha=args.pagerank_alpha,
            pagerank_iters=args.pagerank_iters,
        )

        elapsed_seconds = time.time() - start_time
        cpu_time_end = get_process_cpu_time_seconds(process)
        cpu_time_seconds = None
        if cpu_time_start is not None and cpu_time_end is not None:
            cpu_time_seconds = cpu_time_end - cpu_time_start

        memory_mb = None
        memory_percent = None
        if process is not None:
            memory_info = process.memory_info()
            memory_mb = float(memory_info.rss / (1024 * 1024))
            memory_percent = float(process.memory_percent())

        row = make_iteration_row(
            iteration=iteration + 1,
            random_state=iteration_seed,
            train_size=len(train_graphs),
            test_size=len(test_graphs),
            accuracy=float(metrics["accuracy"]),
            auroc=float(metrics["roc_auc"]),
            balanced_accuracy=float(metrics["balanced_accuracy"]),
            f1=float(metrics["f1"]),
            precision=float(metrics["precision"]),
            recall=float(metrics["recall"]),
            confusion=metrics["confusion_matrix"],
            cpu_time_seconds=cpu_time_seconds,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            elapsed_seconds=elapsed_seconds,
        )
        metrics_rows.append(row)
        confusion_matrices.append(metrics["confusion_matrix"])
        training_seconds_list.append(training_seconds)
        testing_seconds_list.append(testing_seconds)

        iteration_auroc = float(metrics["roc_auc"])
        if iteration_auroc > best_auroc:
            best_auroc = iteration_auroc
            best_iteration_predictions = predictions
            best_iteration_record = row
        if iteration_auroc < worst_auroc:
            worst_auroc = iteration_auroc
            worst_iteration_record = row

        print(
            f"Iteracion {iteration + 1}/{args.iterations} | "
            f"acc={metrics['accuracy']:.4f} auroc={metrics['roc_auc']:.4f} "
            f"bacc={metrics['balanced_accuracy']:.4f} f1={metrics['f1']:.4f}"
        )

    if best_iteration_predictions is None or best_iteration_record is None or worst_iteration_record is None:
        raise RuntimeError("No se pudo determinar una mejor iteracion para exportar predicciones.")

    average_metrics = {
        "Accuracy": float(np.mean([float(row["Accuracy"]) for row in metrics_rows])),
        "AUROC": float(np.mean([float(row["AUROC"]) for row in metrics_rows])),
        "Balanced_Accuracy": float(np.mean([float(row["Balanced_Accuracy"]) for row in metrics_rows])),
        "F1_Score": float(np.mean([float(row["F1_Score"]) for row in metrics_rows])),
        "Precision": float(np.mean([float(row["Precision"]) for row in metrics_rows])),
        "Recall": float(np.mean([float(row["Recall"]) for row in metrics_rows])),
        "Confusion_Matrix": average_confusion_matrix(confusion_matrices),
    }
    average_resources = {
        "CPU_Total_Time_Seconds": float(np.mean([float(row["CPU_Total_Time_Seconds"] or 0.0) for row in metrics_rows])),
        "Memory_Usage_%": float(np.mean([float(row["Memory_Usage_%"] or 0.0) for row in metrics_rows])),
        "Memory_MB": float(np.mean([float(row["Memory_MB"] or 0.0) for row in metrics_rows])),
        "Peak_Memory_MB": float(max(float(row["Memory_MB"] or 0.0) for row in metrics_rows)),
    }
    timing_summary = {
        "Elapsed_Seconds_Total": float(time.time() - overall_start),
        "Elapsed_Seconds_Average": float(np.mean([float(row["Elapsed_Seconds"]) for row in metrics_rows])),
        "Training_Seconds_Total": float(sum(training_seconds_list)),
        "Testing_Seconds_Total": float(sum(testing_seconds_list)),
    }

    metrics_payload = {
        "pipeline": f"graphHD_{dataset_stem}",
        "dataset_path": args.dataset_csv,
        "iterations": args.iterations,
        "random_state_base": args.random_state,
        "test_fraction": args.test_size,
        "target_column": args.target_column,
        "smiles_column": args.smiles_column,
        "id_column": args.id_column,
        "dim": args.dim,
        "pagerank_alpha": args.pagerank_alpha,
        "pagerank_iters": args.pagerank_iters,
        "total_rows_original": int(len(graphs) + invalid_rows),
        "total_rows_valid_rdkit": int(len(graphs)),
        "invalid_rows": int(invalid_rows),
        "max_graph_size": int(max_graph_size),
        "average_metrics": average_metrics,
        "best_run": best_iteration_record,
        "worst_run": worst_iteration_record,
        "average_resources": average_resources,
        "timing_summary": timing_summary,
        "average_confusion_matrix": average_metrics["Confusion_Matrix"],
    }

    metrics_json = output_dir / f"{dataset_stem}_graphhd_metrics.json"
    detailed_csv = output_dir / f"{dataset_stem}_graphhd_detailed_results.csv"
    predictions_csv = output_dir / f"{dataset_stem}_graphhd_best_iteration_predictions.csv"

    write_json(metrics_json, metrics_payload)
    write_csv(detailed_csv, metrics_rows)
    write_csv(predictions_csv, best_iteration_predictions)

    print()
    print_stats_block("Stats corresponding to Maximum AUROC are: ", best_iteration_record)
    print_stats_block("Stats corresponding to Minimum AUROC are: ", worst_iteration_record)
    print(f"Average Stats for {args.iterations} iterations")
    print("Accuracy: ", average_metrics["Accuracy"])
    print("Auroc: ", average_metrics["AUROC"])
    print("Bacc: ", average_metrics["Balanced_Accuracy"])
    print("F1: ", average_metrics["F1_Score"])
    print("Precision: ", average_metrics["Precision"])
    print("Recall: ", average_metrics["Recall"])
    print("Average CPU Total Time: {:.2f}s".format(average_resources["CPU_Total_Time_Seconds"]))
    print("Average Memory Usage: {:.2f}% ({:.2f} MB)".format(average_resources["Memory_Usage_%"], average_resources["Memory_MB"]))
    print("Peak Memory Usage: {:.2f} MB".format(average_resources["Peak_Memory_MB"]))
    print()
    print("Tiempos totales:")
    print("Tiempo total: {:.2f}s".format(timing_summary["Elapsed_Seconds_Total"]))
    print("Tiempo total de entrenamiento: {:.2f}s".format(timing_summary["Training_Seconds_Total"]))
    print("Tiempo total de testeo: {:.2f}s".format(timing_summary["Testing_Seconds_Total"]))
    print()
    print(f"Metricas guardadas en: {metrics_json}")
    print(f"Resultados detallados guardados en: {detailed_csv}")
    print(f"Predicciones mejor iteracion guardadas en: {predictions_csv}")


if __name__ == "__main__":
    main()
