from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, MLP, global_add_pool

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


DEFAULT_BINARY_DATASET_CSV = Path("/tesis/Tesis-PolymerHDC/data/bbbp/raw/BBBP.csv")
DEFAULT_BATCH_SIZE = 64
DEFAULT_HIDDEN_CHANNELS = 64
DEFAULT_NUM_LAYERS = 4
DEFAULT_DROPOUT = 0.2
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 40
DEFAULT_RANDOM_STATE = 800
DEFAULT_ITERATIONS = 100
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.1
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_MIN_EPOCHS = 8
DEFAULT_MIN_DELTA = 1e-4

HYBRIDIZATION_TO_INDEX = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
}
BOND_TYPE_TO_INDEX = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}

RDLogger.DisableLog("rdApp.*")


class PolymerGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int = 4,
        global_dim: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        self.edge_encoder = nn.Linear(edge_dim, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([hidden_channels, hidden_channels, hidden_channels], norm=None)
            self.convs.append(GINEConv(nn=mlp, edge_dim=hidden_channels))
        self.dropout = nn.Dropout(dropout)
        self.classifier = MLP(
            [hidden_channels + global_dim, hidden_channels, num_classes],
            norm=None,
            dropout=dropout,
        )

    def forward(self, x, edge_index, edge_attr, batch, u):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr).relu()
            x = self.dropout(x)
        pooled = global_add_pool(x, batch)
        return self.classifier(torch.cat([pooled, u], dim=-1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Polymer GNN runner")
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_BINARY_DATASET_CSV)
    parser.add_argument(
        "--tu-graph-npz",
        default=None,
        help=(
            "Ruta a un .npz con el grafo real de un benchmark TU Dataset (mutag/ptc_fm/nci1), "
            "generado por dataset_preparation.py (adapter tu_dataset_graph). Si se pasa (solo "
            "con --task binary), se ignoran --dataset-csv/--smiles-column/--target-column: los "
            "grafos se arman directo desde node_labels/edge_labels/edge_index/graph_indicator, "
            "sin pasar nunca por SMILES ni RDKit."
        ),
    )
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--target-column", default="p_np")
    parser.add_argument("--id-column", default="num")
    parser.add_argument("--output-dir", type=Path, default=Path("/run_outputs/artifacts"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--hidden-channels", type=int, default=DEFAULT_HIDDEN_CHANNELS)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--early-stopping-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--min-epochs", type=int, default=DEFAULT_MIN_EPOCHS)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Nucleos a usar para construir grafos moleculares y cargar batches. "
            "0 (default) usa todos los nucleos disponibles."
        ),
    )
    return parser.parse_args()


def resolve_num_workers(num_workers: int) -> int:
    if num_workers > 0:
        return num_workers
    return os.cpu_count() or 1


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atom_features(atom: Chem.Atom) -> list[float]:
    hybridization = [0.0] * len(HYBRIDIZATION_TO_INDEX)
    hybrid_idx = HYBRIDIZATION_TO_INDEX.get(atom.GetHybridization())
    if hybrid_idx is not None:
        hybridization[hybrid_idx] = 1.0
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetTotalDegree() / 6.0,
        atom.GetFormalCharge() / 4.0,
        float(atom.GetIsAromatic()),
        atom.GetTotalNumHs(includeNeighbors=True) / 4.0,
        atom.GetMass() / 200.0,
        float(atom.IsInRing()),
        *hybridization,
    ]


def bond_features(bond: Chem.Bond | None, is_virtual: bool) -> list[float]:
    bond_one_hot = [0.0] * len(BOND_TYPE_TO_INDEX)
    if bond is not None:
        bond_idx = BOND_TYPE_TO_INDEX.get(bond.GetBondType())
        if bond_idx is not None:
            bond_one_hot[bond_idx] = 1.0
    return [
        *bond_one_hot,
        float(bond.GetIsConjugated()) if bond is not None else 0.0,
        float(bond.IsInRing()) if bond is not None else 0.0,
        float(is_virtual),
    ]


def _build_binary_graph_fields(
    fields: tuple[str, object, str],
) -> tuple[str, str, int, list[list[float]], list[list[int]], list[list[float]], list[float]] | None:
    """Igual que _build_polymer_graph_fields: listas planas, sin torch.Tensor,
    para que el resultado cruce procesos sin tocar la memoria compartida de torch."""
    smiles, target, row_id = fields
    if not smiles or pd.isna(target):
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_features = [atom_features(atom) + [0.0] for atom in mol.GetAtoms()]
    edge_pairs: list[list[int]] = []
    edge_features: list[list[float]] = []
    for bond in mol.GetBonds():
        src = bond.GetBeginAtomIdx()
        dst = bond.GetEndAtomIdx()
        features = bond_features(bond, is_virtual=False)
        edge_pairs.append([src, dst])
        edge_pairs.append([dst, src])
        edge_features.append(features)
        edge_features.append(features)

    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    global_features = [num_atoms / 100.0, num_bonds / 100.0, aromatic_atoms / max(num_atoms, 1)]
    return row_id, smiles, int(target), node_features, edge_pairs, edge_features, global_features


def _binary_fields_to_data(
    fields: tuple[str, str, int, list[list[float]], list[list[int]], list[list[float]], list[float]],
) -> Data:
    row_id, smiles, target, node_features, edge_pairs, edge_features, global_features = fields
    empty_bond_dim = len(bond_features(None, is_virtual=False))
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous() if edge_pairs else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else torch.empty((0, empty_bond_dim), dtype=torch.float)
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([target], dtype=torch.long),
        u=torch.tensor([global_features], dtype=torch.float),
    )
    data.row_id = row_id
    data.smiles = smiles
    return data


def build_binary_graphs(
    dataset_csv: Path,
    smiles_column: str,
    target_column: str,
    id_column: str,
    num_workers: int = 1,
) -> tuple[list[Data], int]:
    dataframe = pd.read_csv(dataset_csv)
    required = {smiles_column, target_column, id_column}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {dataset_csv}: {missing}")

    row_fields = [
        (str(row[smiles_column]).strip(), row[target_column], str(row[id_column]).strip())
        for row in dataframe.to_dict("records")
    ]
    del dataframe

    # Cada resultado se tensoriza apenas llega del worker, consumiendo el
    # iterador de executor.map en vez de list(). Juntando primero todos los
    # `raw_results` (listas de floats de Python, con el overhead de objeto de
    # cada float) y tensorizando despues, las dos representaciones del dataset
    # entero quedaban vivas al mismo tiempo: ese era el pico de memoria del
    # armado en los datasets grandes. El orden de los grafos no cambia --
    # executor.map preserva el orden de entrada.
    graphs: list[Data] = []
    processed_rows = 0

    if num_workers > 1 and len(row_fields) > 1:
        chunksize = max(1, len(row_fields) // (num_workers * 4))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for fields in executor.map(_build_binary_graph_fields, row_fields, chunksize=chunksize):
                processed_rows += 1
                if fields is not None:
                    graphs.append(_binary_fields_to_data(fields))
    else:
        for row_field in row_fields:
            fields = _build_binary_graph_fields(row_field)
            processed_rows += 1
            if fields is not None:
                graphs.append(_binary_fields_to_data(fields))

    invalid_rows = processed_rows - len(graphs)

    if not graphs:
        raise RuntimeError("No quedaron moleculas validas para el GNN binario.")
    return graphs, invalid_rows


def build_tu_graphs(npz_path: str) -> tuple[list[Data], int]:
    """Arma los Data de PyG directo desde el .npz de un benchmark TU Dataset
    (mutag/ptc_fm/nci1, generado por dataset_preparation.py), sin SMILES ni
    RDKit: node_labels/edge_labels son categoricos (no hay quimica real de
    por medio en estos benchmarks), asi que se codifican one-hot en vez de
    usar atom_features/bond_features. `u` (features globales) usa el mismo
    esquema simple que build_binary_graphs (conteos de nodos/aristas
    normalizados), con 0.0 en el lugar de la fraccion aromatica ya que ese
    concepto no aplica a un grafo TU abstracto."""
    data = np.load(npz_path)
    edge_index_all = data["edge_index"]
    graph_indicator = data["graph_indicator"]
    graph_labels = data["graph_labels"]
    node_labels = data["node_labels"]
    edge_labels = data["edge_labels"] if "edge_labels" in data.files else None

    num_graphs = int(graph_labels.shape[0])
    num_node_classes = int(node_labels.max()) + 1
    num_edge_classes = int(edge_labels.max()) + 1 if edge_labels is not None else 1

    node_counts = np.bincount(graph_indicator, minlength=num_graphs)
    node_offsets = np.concatenate([[0], np.cumsum(node_counts)])
    edge_graph_ids = graph_indicator[edge_index_all[0]]

    graphs: list[Data] = []
    for graph_id in range(num_graphs):
        num_nodes = int(node_counts[graph_id])
        if num_nodes == 0:
            continue

        start, end = int(node_offsets[graph_id]), int(node_offsets[graph_id + 1])
        local_node_labels = torch.from_numpy(node_labels[start:end]).long()
        x = torch.zeros((num_nodes, num_node_classes), dtype=torch.float)
        x[torch.arange(num_nodes), local_node_labels] = 1.0

        edge_mask = edge_graph_ids == graph_id
        local_edge_index = torch.from_numpy(edge_index_all[:, edge_mask] - node_offsets[graph_id]).long()
        num_edges = int(local_edge_index.size(1))

        if edge_labels is not None:
            local_edge_labels = torch.from_numpy(edge_labels[edge_mask]).long()
            edge_attr = torch.zeros((num_edges, num_edge_classes), dtype=torch.float)
            if num_edges > 0:
                edge_attr[torch.arange(num_edges), local_edge_labels] = 1.0
        else:
            edge_attr = torch.zeros((num_edges, num_edge_classes), dtype=torch.float)

        global_features = [num_nodes / 100.0, num_edges / 100.0, 0.0]

        graph_data = Data(
            x=x,
            edge_index=local_edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([int(graph_labels[graph_id])], dtype=torch.long),
            u=torch.tensor([global_features], dtype=torch.float),
        )
        graph_data.row_id = str(graph_id)
        graph_data.smiles = ""
        graphs.append(graph_data)

    if not graphs:
        raise RuntimeError(f"No quedaron grafos validos al leer {npz_path}.")
    return graphs, 0


def compute_class_weights(train_graphs, num_classes: int, device) -> torch.Tensor:
    """Pesos estilo sklearn class_weight='balanced': n_samples / (n_classes * count_por_clase)."""
    labels = np.asarray([int(graph.y.item()) for graph in train_graphs], dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float, device=device)


def train_one_epoch(model, loader, optimizer, device, class_weights: torch.Tensor | None = None) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.u)
        loss = F.cross_entropy(logits, batch.y, weight=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def collect_scores(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_score = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.u)
        y_true.append(batch.y.cpu())
        y_score.append(torch.softmax(logits, dim=-1).cpu())
    return torch.cat(y_true).numpy(), torch.cat(y_score).numpy()


@torch.no_grad()
def collect_binary_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    model.eval()
    y_true_batches = []
    y_score_batches = []
    rows: list[dict[str, object]] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.u)
        probabilities = torch.softmax(logits, dim=-1).cpu()
        labels = batch.y.cpu()
        y_true_batches.append(labels)
        y_score_batches.append(probabilities)
        for idx in range(labels.size(0)):
            rows.append(
                {
                    "id": batch.row_id[idx],
                    "smiles": batch.smiles[idx],
                    "y_true": int(labels[idx]),
                    "y_score_positive": float(probabilities[idx, 1]),
                    "y_pred": int(probabilities[idx].argmax()),
                }
            )
    return torch.cat(y_true_batches).numpy(), torch.cat(y_score_batches).numpy(), rows



def binary_metrics_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, object]:
    y_pred = y_score.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score[:, 1])),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def average_confusion_matrix(confusion_matrices: list[list[list[int]]]) -> list[list[float]]:
    return np.asarray(confusion_matrices, dtype=np.float64).mean(axis=0).tolist()


def std_metric(values: list[float]) -> float:
    """Desviacion estandar muestral (ddof=1). Con 1 sola iteracion no hay
    variabilidad que medir, asi que devuelve 0.0 en vez de dividir por 0."""
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


_CANONICAL_FIELD_NAMES = {
    "Iteration": "iteration",
    "Random_State": "random_state",
    "Train_Size": "train_size",
    "Val_Size": "val_size",
    "Test_Size": "test_size",
    "Best_Val_Accuracy": "best_val_accuracy",
    "Accuracy": "accuracy",
    "AUROC": "roc_auc",
    "Balanced_Accuracy": "balanced_accuracy",
    "F1_Score": "f1",
    "Precision": "precision",
    "Recall": "recall",
    "Confusion_Matrix": "confusion_matrix",
    "Overall_Mean_AUROC": "overall_mean_roc_auc",
    "Overall_Mean_Accuracy": "overall_mean_accuracy",
    "Task_Metrics": "task_metrics",
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
    internos (compartidas con los CSV de detalle) al mismo vocabulario
    snake_case que usan random_forest_rdkit, mole_bert_hdc, etc., para que
    metrics.json (y por lo tanto results/all_runs.csv) tenga las mismas
    columnas sin importar el pipeline. No toca los CSV de detalle, que se
    siguen escribiendo con las columnas originales."""
    result: dict[str, object] = {}
    for key, value in source.items():
        canonical_key = _CANONICAL_FIELD_NAMES.get(key, key)
        if canonical_key == "confusion_matrix" and isinstance(value, str):
            value = json.loads(value)
        result[canonical_key] = value
    return result


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No hay filas para escribir en {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_process_cpu_time_seconds(process: "psutil.Process | None") -> float | None:
    if process is None:
        return None
    cpu_times = process.cpu_times()
    return float(cpu_times.user + cpu_times.system)


def build_model(sample: Data, args: argparse.Namespace, num_classes: int) -> PolymerGNN:
    return PolymerGNN(
        in_channels=sample.x.size(-1),
        edge_dim=sample.edge_attr.size(-1),
        hidden_channels=args.hidden_channels,
        num_classes=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        global_dim=sample.u.size(-1),
    )


def run_binary(args: argparse.Namespace) -> None:
    overall_start = time.time()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    num_workers = resolve_num_workers(args.num_workers)

    if args.tu_graph_npz:
        graphs, invalid_rows = build_tu_graphs(args.tu_graph_npz)
        dataset_stem = Path(args.tu_graph_npz).stem.lower().removesuffix("_graph")
        dataset_display = args.tu_graph_npz
        dataset_path_for_summary = args.tu_graph_npz
    else:
        dataset_csv = args.dataset_csv
        if not dataset_csv.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_csv}")
        graphs, invalid_rows = build_binary_graphs(
            dataset_csv, args.smiles_column, args.target_column, args.id_column, num_workers=num_workers
        )
        dataset_stem = dataset_csv.stem.lower()
        dataset_display = str(dataset_csv)
        dataset_path_for_summary = str(dataset_csv)

    process = psutil.Process() if psutil is not None else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(num_workers)
    sample = graphs[0]
    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    confusion_matrices: list[list[list[int]]] = []
    best_iteration_payload = None
    worst_iteration_payload = None
    best_auroc = float("-inf")
    worst_auroc = float("inf")

    print(f"Loaded {len(graphs)} valid graphs from {dataset_display}")
    print(f"Invalid rows skipped: {invalid_rows}")
    print(f"Iteraciones: {args.iterations}")
    print(f"Epochs maximos por iteracion: {args.epochs}")

    # Las etiquetas y los indices no dependen de la iteracion: se calculaban
    # de cero en cada una de las 100 vueltas, recorriendo los N grafos.
    labels = np.asarray([int(graph.y.item()) for graph in graphs], dtype=np.int64)
    indices = np.arange(len(graphs))

    for iteration in range(args.iterations):
        iteration_seed = args.random_state + iteration
        set_seed(iteration_seed)
        cpu_time_start = get_process_cpu_time_seconds(process)
        start_time = time.time()

        train_indices, test_indices = train_test_split(
            indices,
            test_size=args.test_size,
            random_state=iteration_seed,
            stratify=labels,
        )
        val_fraction_of_train = args.val_size / max(1.0 - args.test_size, 1e-8)
        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=val_fraction_of_train,
            random_state=iteration_seed,
            stratify=labels[train_indices],
        )

        train_graphs = [graphs[idx] for idx in train_indices]
        val_graphs = [graphs[idx] for idx in val_indices]
        test_graphs = [graphs[idx] for idx in test_indices]
        train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
        test_loader = DataLoader(test_graphs, batch_size=args.batch_size)
        model = build_model(sample, args, num_classes=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        class_weights = compute_class_weights(train_graphs, num_classes=2, device=device)

        best_state = None
        best_val_accuracy = float("-inf")
        epochs_without_improvement = 0
        for epoch in range(1, args.epochs + 1):
            train_one_epoch(model, train_loader, optimizer, device, class_weights=class_weights)
            val_metrics = binary_metrics_from_scores(*collect_scores(model, val_loader, device))
            current_val_accuracy = float(val_metrics["accuracy"])
            if current_val_accuracy > best_val_accuracy + args.min_delta:
                best_val_accuracy = current_val_accuracy
                best_state = {key: value.cpu() for key, value in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epoch >= args.min_epochs and epochs_without_improvement >= args.early_stopping_patience:
                break

        if best_state is None:
            raise RuntimeError("Training did not produce a checkpoint")

        training_seconds = float(time.time() - start_time)
        testing_start = time.time()
        model.load_state_dict(best_state)
        y_true, y_score, prediction_batch_rows = collect_binary_predictions(model, test_loader, device)
        test_metrics = binary_metrics_from_scores(y_true, y_score)
        testing_seconds = float(time.time() - testing_start)
        elapsed_seconds = training_seconds + testing_seconds
        cpu_time_seconds = None if process is None or cpu_time_start is None else max(0.0, get_process_cpu_time_seconds(process) - cpu_time_start)
        memory_percent = None if process is None else float(process.memory_percent())
        memory_mb = None if process is None else float(process.memory_info().rss / 1024 / 1024)

        row = {
            "Iteration": iteration + 1,
            "Random_State": iteration_seed,
            "Train_Size": len(train_graphs),
            "Val_Size": len(val_graphs),
            "Test_Size": len(test_graphs),
            "Best_Val_Accuracy": best_val_accuracy,
            "Accuracy": test_metrics["accuracy"],
            "AUROC": test_metrics["roc_auc"],
            "Balanced_Accuracy": test_metrics["balanced_accuracy"],
            "F1_Score": test_metrics["f1"],
            "Precision": test_metrics["precision"],
            "Recall": test_metrics["recall"],
            "Confusion_Matrix": json.dumps(test_metrics["confusion_matrix"], ensure_ascii=False),
            "CPU_Total_Time_Seconds": cpu_time_seconds,
            "Memory_Usage_%": memory_percent,
            "Memory_MB": memory_mb,
            "Elapsed_Seconds": elapsed_seconds,
            "Training_Seconds": training_seconds,
            "Testing_Seconds": testing_seconds,
        }
        rows.append(row)
        confusion_matrices.append(test_metrics["confusion_matrix"])
        print(
            f"[{iteration + 1}/{args.iterations}] accuracy={row['Accuracy']:.4f} "
            f"auroc={row['AUROC']:.4f} balanced_accuracy={row['Balanced_Accuracy']:.4f} "
            f"f1={row['F1_Score']:.4f} precision={row['Precision']:.4f} recall={row['Recall']:.4f} "
            f"random_state={iteration_seed} "
            f"cpu_time={(cpu_time_seconds or 0.0):.2f}s memory={(memory_mb or 0.0):.1f}MB "
            f"elapsed={elapsed_seconds:.1f}s"
        )

        for prediction_row in prediction_batch_rows:
            prediction_rows.append(
                {
                    "iteration": iteration + 1,
                    "random_state": iteration_seed,
                    **prediction_row,
                }
            )

        if test_metrics["roc_auc"] > best_auroc:
            best_auroc = float(test_metrics["roc_auc"])
            best_iteration_payload = row
        if test_metrics["roc_auc"] < worst_auroc:
            worst_auroc = float(test_metrics["roc_auc"])
            worst_iteration_payload = row

    cpu_values = [row["CPU_Total_Time_Seconds"] for row in rows if row["CPU_Total_Time_Seconds"] is not None]
    memory_percent_values = [row["Memory_Usage_%"] for row in rows if row["Memory_Usage_%"] is not None]
    memory_mb_values = [row["Memory_MB"] for row in rows if row["Memory_MB"] is not None]

    metrics = {
        "pipeline": "polymer_gnn_bbbp",
        "dataset_path": dataset_path_for_summary,
        "total_rows_original": len(graphs) + invalid_rows,
        "total_rows_valid_rdkit": len(graphs),
        "invalid_rows": invalid_rows,
        "iterations": args.iterations,
        "random_state_base": args.random_state,
        "test_fraction": args.test_size,
        "train_size": best_iteration_payload["Train_Size"],
        "test_size": best_iteration_payload["Test_Size"],
        "average_metrics": canonicalize_metric_dict({
            "Accuracy": float(np.mean([row["Accuracy"] for row in rows])),
            "AUROC": float(np.mean([row["AUROC"] for row in rows])),
            "Balanced_Accuracy": float(np.mean([row["Balanced_Accuracy"] for row in rows])),
            "F1_Score": float(np.mean([row["F1_Score"] for row in rows])),
            "Precision": float(np.mean([row["Precision"] for row in rows])),
            "Recall": float(np.mean([row["Recall"] for row in rows])),
            "Confusion_Matrix": average_confusion_matrix(confusion_matrices),
        }),
        "average_resources": {
            "cpu_time_seconds": float(np.mean(cpu_values)) if cpu_values else None,
            "memory_percent": float(np.mean(memory_percent_values)) if memory_percent_values else None,
            "memory_mb": float(np.mean(memory_mb_values)) if memory_mb_values else None,
            "peak_memory_mb": float(max(memory_mb_values)) if memory_mb_values else None,
            "elapsed_seconds": float(np.mean([row["Elapsed_Seconds"] for row in rows])),
        },
        "best_run": canonicalize_metric_dict(best_iteration_payload),
        "worst_run": canonicalize_metric_dict(worst_iteration_payload),
        "timing_summary": {
            "total_wall_clock_seconds": float(time.time() - overall_start),
            "total_training_seconds": float(sum(row["Training_Seconds"] for row in rows)),
            "total_testing_seconds": float(sum(row["Testing_Seconds"] for row in rows)),
        },
    }
    write_json(output_dir / f"{dataset_stem}_polymer_gnn_metrics.json", metrics)
    write_csv(output_dir / f"{dataset_stem}_polymer_gnn_detailed_results.csv", rows)
    write_csv(output_dir / f"{dataset_stem}_polymer_gnn_best_iteration_predictions.csv", prediction_rows)

    average_metrics = metrics["average_metrics"]
    print()
    print(f"Average Stats for {args.iterations} iterations")
    print("Accuracy: ", average_metrics["accuracy"])
    print("Auroc: ", average_metrics["roc_auc"])
    print("Bacc: ", average_metrics["balanced_accuracy"])
    print("F1: ", average_metrics["f1"])
    print("Precision: ", average_metrics["precision"])
    print("Recall: ", average_metrics["recall"])
    print()
    print(f"Standard Deviation for {args.iterations} iterations")
    print("Accuracy: ", std_metric([row["Accuracy"] for row in rows]))
    print("Auroc: ", std_metric([row["AUROC"] for row in rows]))
    print("Bacc: ", std_metric([row["Balanced_Accuracy"] for row in rows]))
    print("F1: ", std_metric([row["F1_Score"] for row in rows]))
    print("Precision: ", std_metric([row["Precision"] for row in rows]))
    print("Recall: ", std_metric([row["Recall"] for row in rows]))


def main() -> None:
    args = parse_args()
    run_binary(args)
