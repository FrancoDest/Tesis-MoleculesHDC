from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
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

# Tiene que correr antes de CUALQUIER operacion de torch en el proceso
# principal (antes de crear tensores, generadores, etc.): si el pool de
# threads OpenMP/MKL de torch llega a inicializarse en el padre y recien
# despues se hace fork() para el ProcessPoolExecutor de encode_all_graphs,
# los hijos heredan ese pool en un estado invalido y quedan colgados (deadlock
# con 0% de CPU) apenas llaman a torchhd.bind/bundle. Con 1 solo thread nunca
# se llega a inicializar ese pool, asi que forkear despues es seguro.
# Confirmado en un repro aislado: sin esto, ProcessPoolExecutor se cuelga en
# la primera tarea; con esto, corre normal.
torch.set_num_threads(1)


@dataclass(frozen=True)
class MoleculeGraph:
    row_id: str
    smiles: str
    label: int
    # numpy en vez de torch.Tensor: al cruzar procesos via ProcessPoolExecutor,
    # un torch.Tensor dispara el reductor de memoria compartida de torch (usa
    # /dev/shm), que en un container Docker suele tener muy poco espacio y
    # revienta con miles de grafos. numpy se pickle por valor, sin ese riesgo.
    edge_index: np.ndarray
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
    parser.add_argument("--dataset-csv", default=None, help="CSV con columnas de smiles y target.")
    parser.add_argument(
        "--tu-graph-npz",
        default=None,
        help=(
            "Ruta a un .npz con el grafo real de un benchmark TU Dataset (mutag/ptc_fm/nci1), "
            "generado por dataset_preparation.py (adapter tu_dataset_graph). Si se pasa, se "
            "ignoran --dataset-csv/--smiles-column/--target-column/--id-column: el grafo se "
            "arma directo desde node_labels/edge_index/graph_indicator/graph_labels, sin pasar "
            "nunca por SMILES ni RDKit."
        ),
    )
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Nucleos a usar para parsear SMILES y codificar hypervectores. "
            "0 (default) usa todos los nucleos disponibles."
        ),
    )
    return parser.parse_args()


def resolve_num_workers(num_workers: int) -> int:
    if num_workers > 0:
        return num_workers
    return os.cpu_count() or 1


def _parse_binary_graph_row(fields: tuple[str, object, str]) -> MoleculeGraph | None:
    smiles, target, row_id = fields
    if not smiles or pd.isna(target):
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    edge_index = mol_to_edge_index(mol)
    return MoleculeGraph(
        row_id=row_id,
        smiles=smiles,
        label=int(target),
        edge_index=edge_index,
        num_nodes=mol.GetNumAtoms(),
    )


def load_binary_graphs(
    dataset_csv: str,
    smiles_column: str,
    target_column: str,
    id_column: str,
    num_workers: int = 1,
) -> tuple[list[MoleculeGraph], int]:
    dataframe = pd.read_csv(dataset_csv)
    required = {smiles_column, target_column, id_column}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {dataset_csv}: {missing}")

    row_fields = [
        (str(row[smiles_column]).strip(), row[target_column], str(row[id_column]).strip())
        for row in dataframe.to_dict("records")
    ]

    if num_workers > 1 and len(row_fields) > 1:
        chunksize = max(1, len(row_fields) // (num_workers * 4))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_parse_binary_graph_row, row_fields, chunksize=chunksize))
    else:
        results = [_parse_binary_graph_row(fields) for fields in row_fields]

    graphs = [graph for graph in results if graph is not None]
    invalid_rows = len(results) - len(graphs)

    if not graphs:
        raise ValueError("No quedaron moleculas validas para GraphHD.")
    return graphs, invalid_rows


def load_tu_graphs(npz_path: str) -> tuple[list[MoleculeGraph], int]:
    """Arma la lista de MoleculeGraph directo desde el .npz de un benchmark TU
    Dataset (mutag/ptc_fm/nci1), sin SMILES ni RDKit de por medio: separa los
    arrays globales (edge_index, graph_indicator) por grafo. node_labels no
    se usa aca -- GraphHD, tal como esta en este repo, solo codifica la
    topologia del grafo (edge_index/num_nodes), igual que en el camino SMILES
    existente (que tampoco usa el tipo de atomo, solo los enlaces)."""
    data = np.load(npz_path)
    edge_index_all = data["edge_index"]
    graph_indicator = data["graph_indicator"]
    graph_labels = data["graph_labels"]

    num_graphs = int(graph_labels.shape[0])
    node_counts = np.bincount(graph_indicator, minlength=num_graphs)
    node_offsets = np.concatenate([[0], np.cumsum(node_counts)])

    edge_graph_ids = graph_indicator[edge_index_all[0]]

    graphs: list[MoleculeGraph] = []
    for graph_id in range(num_graphs):
        num_nodes = int(node_counts[graph_id])
        if num_nodes == 0:
            continue
        edge_mask = edge_graph_ids == graph_id
        local_edge_index = edge_index_all[:, edge_mask] - node_offsets[graph_id]
        graphs.append(
            MoleculeGraph(
                row_id=str(graph_id),
                smiles="",
                label=int(graph_labels[graph_id]),
                edge_index=local_edge_index.astype(np.int64),
                num_nodes=num_nodes,
            )
        )

    if not graphs:
        raise ValueError(f"No quedaron grafos validos al leer {npz_path}.")
    return graphs, 0


def mol_to_edge_index(mol: Chem.Mol) -> np.ndarray:
    edges: list[list[int]] = []
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        edges.append([begin_idx, end_idx])
        edges.append([end_idx, begin_idx])
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T.copy()


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


def encode_graph_with_node_ids(
    graph: MoleculeGraph, node_ids: torch.Tensor, vector_size: int, alpha: float, max_iter: int
) -> torch.Tensor:
    if graph.num_nodes == 0:
        # Vector neutro del modelo BSC (no un zeros() plano): tiene que poder
        # bundlearse con hipervectores de grafos reales sin romper el tipo.
        return torchhd.identity(1, vector_size, vsa="BSC")[0].to(torch.float32)

    # graph.edge_index vive como numpy (ver MoleculeGraph); se convierte a
    # tensor recien aca, localmente, para no cruzar torch.Tensor por IPC.
    torch_graph = replace(graph, edge_index=torch.from_numpy(graph.edge_index))
    pr = pagerank(torch_graph, alpha=alpha, max_iter=max_iter)
    pr_argsort = inverse_permutation(torch.argsort(pr))
    node_id_hvs = node_ids[pr_argsort]
    undirected = to_undirected_unique(torch_graph.edge_index)
    if undirected.numel() == 0:
        return torchhd.identity(1, vector_size, vsa="BSC")[0].to(torch.float32)

    row, col = undirected
    edge_hvs = [torchhd.bind(node_id_hvs[s], node_id_hvs[t]) for s, t in zip(row, col)]
    graph_hv = reduce(torchhd.bundle, edge_hvs)
    return graph_hv.to(torch.float32)


class GraphHDEncoder:
    def __init__(self, num_vectors: int, vector_size: int):
        # node_ids tiene que ser un VSATensor real (no un torch.Tensor plano de
        # 0/1): torchhd.bundle/hamming_similarity dispatchan segun el tipo, y
        # sobre un tensor plano hacen suma sin acotar en vez de voto de mayoria,
        # lo que rompe la comparacion de similitud (ver investigacion del bug
        # de AUROC=0.5 constante).
        self.node_ids = torchhd.random(num_vectors, vector_size, vsa="BSC")
        self.vector_size = vector_size

    def encode(self, graph: MoleculeGraph, alpha: float, max_iter: int) -> torch.Tensor:
        return encode_graph_with_node_ids(graph, self.node_ids, self.vector_size, alpha=alpha, max_iter=max_iter)


_ENCODE_WORKER_NODE_IDS: torch.Tensor | None = None
_ENCODE_WORKER_VECTOR_SIZE: int | None = None


def _init_encode_worker(node_ids_np: np.ndarray, vector_size: int) -> None:
    # node_ids_np llega como numpy (ver encode_all_graphs) para no disparar
    # el reductor de memoria compartida de torch al arrancar cada worker.
    # torch.from_numpy por si solo devuelve un tensor plano: hay que
    # reenvolverlo como VSATensor BSC o se pierde el tipo y bundle/similarity
    # vuelven a comportarse mal (mismo bug que node_ids en GraphHDEncoder).
    global _ENCODE_WORKER_NODE_IDS, _ENCODE_WORKER_VECTOR_SIZE
    _ENCODE_WORKER_NODE_IDS = torchhd.ensure_vsa_tensor(node_ids_np, vsa="BSC")
    _ENCODE_WORKER_VECTOR_SIZE = vector_size


def _encode_graph_worker(args: tuple[MoleculeGraph, float, int]) -> np.ndarray:
    graph, alpha, max_iter = args
    assert _ENCODE_WORKER_NODE_IDS is not None and _ENCODE_WORKER_VECTOR_SIZE is not None
    hv = encode_graph_with_node_ids(
        graph, _ENCODE_WORKER_NODE_IDS, _ENCODE_WORKER_VECTOR_SIZE, alpha=alpha, max_iter=max_iter
    )
    # Se devuelve numpy (no torch.Tensor) para evitar que cada resultado
    # dispare la memoria compartida de torch (/dev/shm) al volver al proceso
    # principal; con miles de grafos eso agotaba el shm del container Docker.
    return hv.numpy()


def encode_all_graphs(
    encoder: GraphHDEncoder,
    graphs: Sequence[MoleculeGraph],
    alpha: float,
    max_iter: int,
    num_workers: int = 1,
) -> list[torch.Tensor]:
    if num_workers > 1 and len(graphs) > 1:
        chunksize = max(1, len(graphs) // (num_workers * 4))
        tasks = [(graph, alpha, max_iter) for graph in graphs]
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_encode_worker,
            initargs=(encoder.node_ids.numpy(), encoder.vector_size),
        ) as executor:
            results = list(executor.map(_encode_graph_worker, tasks, chunksize=chunksize))
        # torch.from_numpy por si solo da un tensor plano: hay que reenvolverlo
        # como VSATensor BSC, si no torchhd.bundle() lo autoconvierte al
        # modelo MAP (bipolar) por default y el bundling con las clases
        # (que si son BSC) queda inconsistente.
        return [torchhd.ensure_vsa_tensor(arr, vsa="BSC") for arr in results]
    return [encoder.encode(graph, alpha=alpha, max_iter=max_iter) for graph in graphs]


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


def std_metric(values: Sequence[float]) -> float:
    """Desviacion estandar muestral (ddof=1). Con 1 sola iteracion no hay
    variabilidad que medir, asi que devuelve 0.0 en vez de dividir por 0."""
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


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


_CANONICAL_FIELD_NAMES = {
    "Iteration": "iteration",
    "Random_State": "random_state",
    "Train_Size": "train_size",
    "Test_Size": "test_size",
    "Accuracy": "accuracy",
    "AUROC": "roc_auc",
    "Balanced_Accuracy": "balanced_accuracy",
    "F1_Score": "f1",
    "Precision": "precision",
    "Recall": "recall",
    "Confusion_Matrix": "confusion_matrix",
    "CPU_Total_Time_Seconds": "cpu_time_seconds",
    "Memory_Usage_%": "memory_percent",
    "Memory_MB": "memory_mb",
    "Peak_Memory_MB": "peak_memory_mb",
    "Elapsed_Seconds": "elapsed_seconds",
    "Training_Seconds": "training_seconds",
    "Testing_Seconds": "testing_seconds",
}


def canonicalize_metric_dict(source: dict[str, object]) -> dict[str, object]:
    """Traduce las claves PascalCase_Con_Guiones de las filas/diccionarios
    internos (compartidas con detailed_results.csv) al mismo vocabulario
    snake_case que usan random_forest_rdkit, mole_bert_hdc, etc., para que
    metrics.json (y por lo tanto results/all_runs.csv) tenga las mismas
    columnas sin importar el pipeline. No toca el CSV de detalle, que se
    sigue escribiendo con las columnas originales."""
    result: dict[str, object] = {}
    for key, value in source.items():
        canonical_key = _CANONICAL_FIELD_NAMES.get(key, key)
        if canonical_key == "confusion_matrix" and isinstance(value, str):
            value = json.loads(value)
        result[canonical_key] = value
    return result


def evaluate_iteration(
    train_hvs: Sequence[torch.Tensor],
    train_labels: Sequence[int],
    test_hvs: Sequence[torch.Tensor],
    test_labels: Sequence[int],
    test_ids: Sequence[str],
    test_smiles: Sequence[str],
    classes: Sequence[int],
) -> tuple[dict[str, float | list[list[int]]], list[dict[str, object]], float, float]:
    training_start = time.time()
    classifier = MajorityClassification()
    for hv, label in zip(train_hvs, train_labels):
        classifier.add(hv, label)
    training_seconds = time.time() - training_start

    y_true: list[int] = []
    y_pred: list[int] = []
    positive_scores: list[float] = []
    prediction_rows: list[dict[str, object]] = []
    positive_class = max(classes)
    positive_index = classifier.class_to_index[positive_class]
    testing_start = time.time()

    for hv, label, row_id, smiles in zip(test_hvs, test_labels, test_ids, test_smiles):
        scores = classifier.score(hv)
        prediction = classifier.predict(hv)
        positive_score = float(scores[positive_index].item())
        y_true.append(label)
        y_pred.append(prediction)
        positive_scores.append(positive_score)
        prediction_rows.append(
            {
                "id": row_id,
                "smiles": smiles,
                "y_true": label,
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
    num_workers = resolve_num_workers(args.num_workers)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.random_state)

    if args.tu_graph_npz:
        print(f"Usando grafo real TU Dataset desde {args.tu_graph_npz} (sin SMILES/RDKit).")
        graphs, invalid_rows = load_tu_graphs(args.tu_graph_npz)
        dataset_stem = Path(args.tu_graph_npz).stem.lower().removesuffix("_graph")
        dataset_path_for_summary = args.tu_graph_npz
    elif args.dataset_csv:
        print(f"Usando {num_workers} nucleos para parseo de SMILES y encoding de hypervectores.")
        graphs, invalid_rows = load_binary_graphs(
            dataset_csv=args.dataset_csv,
            smiles_column=args.smiles_column,
            target_column=args.target_column,
            id_column=args.id_column,
            num_workers=num_workers,
        )
        dataset_stem = Path(args.dataset_csv).stem.lower()
        dataset_path_for_summary = args.dataset_csv
    else:
        raise ValueError("Hay que pasar --dataset-csv o --tu-graph-npz.")

    labels = np.asarray([graph.label for graph in graphs], dtype=np.int64)
    classes = sorted({int(label) for label in labels.tolist()})
    class_counts = {label: int((labels == label).sum()) for label in classes}
    max_graph_size = max(graph.num_nodes for graph in graphs)
    process = psutil.Process() if psutil is not None else None
    encoder = GraphHDEncoder(num_vectors=max_graph_size, vector_size=args.dim)

    # El hypervector de cada grafo no depende de la iteracion (mismo encoder,
    # mismo alpha/max_iter), asi que se calcula una unica vez en vez de
    # recodificar train/test en cada uno de los splits repetidos.
    encoding_start = time.time()
    graph_hvs = encode_all_graphs(
        encoder=encoder,
        graphs=graphs,
        alpha=args.pagerank_alpha,
        max_iter=args.pagerank_iters,
        num_workers=num_workers,
    )
    print(f"Hypervectores precalculados para {len(graphs)} moleculas en {time.time() - encoding_start:.2f}s")

    metrics_rows: list[dict[str, object]] = []
    best_iteration_predictions: list[dict[str, object]] | None = None
    best_iteration_record: dict[str, object] | None = None
    worst_iteration_record: dict[str, object] | None = None
    best_auroc = float("-inf")
    worst_auroc = float("inf")
    confusion_matrices: list[list[list[int]]] = []
    training_seconds_list: list[float] = []
    testing_seconds_list: list[float] = []

    print("Usando variante graphHD mas fiel al notebook original...")
    print(f"Clase 0: {class_counts.get(0, 0)}, Clase 1: {class_counts.get(1, 0)}")
    print(f"Filas invalidas descartadas: {invalid_rows}")
    print(f"Iteraciones: {args.iterations}")
    print(f"Split por iteración: 80/20 con seed {args.random_state} + iteración")

    overall_start = time.time()
    all_indices = np.arange(len(graphs))
    for iteration in range(args.iterations):
        iteration_seed = args.random_state + iteration
        train_idx, test_idx = train_test_split(
            all_indices,
            test_size=args.test_size,
            random_state=iteration_seed,
            stratify=labels,
        )
        start_time = time.time()
        cpu_time_start = get_process_cpu_time_seconds(process)

        metrics, predictions, training_seconds, testing_seconds = evaluate_iteration(
            train_hvs=[graph_hvs[i] for i in train_idx],
            train_labels=[graphs[i].label for i in train_idx],
            test_hvs=[graph_hvs[i] for i in test_idx],
            test_labels=[graphs[i].label for i in test_idx],
            test_ids=[graphs[i].row_id for i in test_idx],
            test_smiles=[graphs[i].smiles for i in test_idx],
            classes=classes,
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
            train_size=len(train_idx),
            test_size=len(test_idx),
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
            f"bacc={metrics['balanced_accuracy']:.4f} f1={metrics['f1']:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
            f"random_state={iteration_seed} cpu_time={cpu_time_seconds or 0.0:.2f}s "
            f"memory={memory_mb or 0.0:.1f}MB elapsed={elapsed_seconds:.2f}s"
        )

    if best_iteration_predictions is None or best_iteration_record is None or worst_iteration_record is None:
        raise RuntimeError("No se pudo determinar una mejor iteracion para exportar predicciones.")

    average_metrics = canonicalize_metric_dict({
        "Accuracy": float(np.mean([float(row["Accuracy"]) for row in metrics_rows])),
        "AUROC": float(np.mean([float(row["AUROC"]) for row in metrics_rows])),
        "Balanced_Accuracy": float(np.mean([float(row["Balanced_Accuracy"]) for row in metrics_rows])),
        "F1_Score": float(np.mean([float(row["F1_Score"]) for row in metrics_rows])),
        "Precision": float(np.mean([float(row["Precision"]) for row in metrics_rows])),
        "Recall": float(np.mean([float(row["Recall"]) for row in metrics_rows])),
        "Confusion_Matrix": average_confusion_matrix(confusion_matrices),
    })
    std_metrics = canonicalize_metric_dict({
        "Accuracy": std_metric([float(row["Accuracy"]) for row in metrics_rows]),
        "AUROC": std_metric([float(row["AUROC"]) for row in metrics_rows]),
        "Balanced_Accuracy": std_metric([float(row["Balanced_Accuracy"]) for row in metrics_rows]),
        "F1_Score": std_metric([float(row["F1_Score"]) for row in metrics_rows]),
        "Precision": std_metric([float(row["Precision"]) for row in metrics_rows]),
        "Recall": std_metric([float(row["Recall"]) for row in metrics_rows]),
    })
    average_resources = {
        "cpu_time_seconds": float(np.mean([float(row["CPU_Total_Time_Seconds"] or 0.0) for row in metrics_rows])),
        "memory_percent": float(np.mean([float(row["Memory_Usage_%"] or 0.0) for row in metrics_rows])),
        "memory_mb": float(np.mean([float(row["Memory_MB"] or 0.0) for row in metrics_rows])),
        "peak_memory_mb": float(max(float(row["Memory_MB"] or 0.0) for row in metrics_rows)),
        "elapsed_seconds": float(np.mean([float(row["Elapsed_Seconds"]) for row in metrics_rows])),
    }
    timing_summary = {
        "total_wall_clock_seconds": float(time.time() - overall_start),
        "total_training_seconds": float(sum(training_seconds_list)),
        "total_testing_seconds": float(sum(testing_seconds_list)),
    }

    metrics_payload = {
        "pipeline": "graphHD",
        "dataset_path": dataset_path_for_summary,
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
        "train_size": best_iteration_record["Train_Size"],
        "test_size": best_iteration_record["Test_Size"],
        "average_metrics": average_metrics,
        "best_run": canonicalize_metric_dict(best_iteration_record),
        "worst_run": canonicalize_metric_dict(worst_iteration_record),
        "average_resources": average_resources,
        "timing_summary": timing_summary,
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
    print("Accuracy: ", average_metrics["accuracy"])
    print("Auroc: ", average_metrics["roc_auc"])
    print("Bacc: ", average_metrics["balanced_accuracy"])
    print("F1: ", average_metrics["f1"])
    print("Precision: ", average_metrics["precision"])
    print("Recall: ", average_metrics["recall"])
    print()
    print(f"Standard Deviation for {args.iterations} iterations")
    print("Accuracy: ", std_metrics["accuracy"])
    print("Auroc: ", std_metrics["roc_auc"])
    print("Bacc: ", std_metrics["balanced_accuracy"])
    print("F1: ", std_metrics["f1"])
    print("Precision: ", std_metrics["precision"])
    print("Recall: ", std_metrics["recall"])
    print("Average CPU Total Time: {:.2f}s".format(average_resources["cpu_time_seconds"]))
    print("Average Memory Usage: {:.2f}% ({:.2f} MB)".format(average_resources["memory_percent"], average_resources["memory_mb"]))
    print("Peak Memory Usage: {:.2f} MB".format(average_resources["peak_memory_mb"]))
    print()
    print("Tiempos totales:")
    print("Tiempo total: {:.2f}s".format(timing_summary["total_wall_clock_seconds"]))
    print("Tiempo total de entrenamiento: {:.2f}s".format(timing_summary["total_training_seconds"]))
    print("Tiempo total de testeo: {:.2f}s".format(timing_summary["total_testing_seconds"]))
    print()
    print(f"Metricas guardadas en: {metrics_json}")
    print(f"Resultados detallados guardados en: {detailed_csv}")
    print(f"Predicciones mejor iteracion guardadas en: {predictions_csv}")


if __name__ == "__main__":
    main()
