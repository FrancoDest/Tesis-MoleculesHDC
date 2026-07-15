from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from dataclasses import dataclass
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
DEFAULT_VIPEA_CSV = Path("/tesis/Tesis-PolymerHDC/data/vipea/dataset.csv")
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

VIPEA_TARGET_COLUMNS = {"EA": "EA (eV)", "IP": "IP (eV)"}
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


@dataclass(frozen=True)
class PolymerRecord:
    poly_id: str
    poly_type: str
    comp: str
    frac_a: float
    frac_b: float
    mono_a: str
    mono_b: str
    ea: float
    ip: float


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
    parser.add_argument("--task", choices=("binary", "vipea"), required=True)
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_BINARY_DATASET_CSV)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_VIPEA_CSV)
    parser.add_argument(
        "--target-property",
        choices=("EA", "IP"),
        default="EA",
        help="Propiedad de VIPEA a predecir (task=vipea), binarizada por su mediana.",
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
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


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


def load_polymer_records(csv_path: Path, max_samples: int | None = None) -> list[PolymerRecord]:
    records: list[PolymerRecord] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                PolymerRecord(
                    poly_id=row["poly_id"],
                    poly_type=row["poly_type"].strip(),
                    comp=row["comp"].strip(),
                    frac_a=float(row["fracA"]),
                    frac_b=float(row["fracB"]),
                    mono_a=row["monoA"].strip(),
                    mono_b=row["monoB"].strip(),
                    ea=float(row["EA (eV)"]),
                    ip=float(row["IP (eV)"]),
                )
            )
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def parse_comp(comp: str) -> tuple[int, int]:
    left, right = comp.split("_")
    return int(left[:-1]), int(right[:-1])


def make_sequence(record: PolymerRecord, count_a: int, count_b: int) -> list[str]:
    if record.poly_type == "alternating":
        sequence: list[str] = []
        remaining_a = count_a
        remaining_b = count_b
        next_unit = "A"
        while remaining_a > 0 or remaining_b > 0:
            if next_unit == "A" and remaining_a > 0:
                sequence.append("A")
                remaining_a -= 1
            elif next_unit == "B" and remaining_b > 0:
                sequence.append("B")
                remaining_b -= 1
            elif remaining_a > 0:
                sequence.append("A")
                remaining_a -= 1
            elif remaining_b > 0:
                sequence.append("B")
                remaining_b -= 1
            next_unit = "B" if next_unit == "A" else "A"
        return sequence
    if record.poly_type == "block":
        return ["A"] * count_a + ["B"] * count_b
    if record.poly_type == "random":
        sequence = ["A"] * count_a + ["B"] * count_b
        rng = random.Random(stable_seed(record.poly_id, record.comp, record.mono_a, record.mono_b))
        rng.shuffle(sequence)
        return sequence
    raise ValueError(f"Unsupported poly_type: {record.poly_type}")


def pick_anchor_index(mol: Chem.Mol) -> int:
    candidates = [
        (atom.GetDegree(), atom.GetAtomicNum(), -atom.GetFormalCharge(), atom.GetIdx())
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    if not candidates:
        return 0
    return max(candidates)[-1]


def mol_to_fragment(mol: Chem.Mol, unit_tag: float) -> tuple[list[list[float]], list[list[int]], list[list[float]], int]:
    x = [atom_features(atom) + [unit_tag] for atom in mol.GetAtoms()]
    edge_index: list[list[int]] = []
    edge_attr: list[list[float]] = []
    for bond in mol.GetBonds():
        src = bond.GetBeginAtomIdx()
        dst = bond.GetEndAtomIdx()
        features = bond_features(bond, is_virtual=False)
        edge_index.append([src, dst])
        edge_index.append([dst, src])
        edge_attr.append(features)
        edge_attr.append(features)
    anchor_idx = pick_anchor_index(mol)
    return x, edge_index, edge_attr, anchor_idx


def build_polymer_graph(record: PolymerRecord, target_property: str, threshold: float) -> Data:
    count_a, count_b = parse_comp(record.comp)
    sequence = make_sequence(record, count_a, count_b)
    mol_a = Chem.MolFromSmiles(record.mono_a)
    mol_b = Chem.MolFromSmiles(record.mono_b)
    if mol_a is None or mol_b is None:
        raise ValueError(f"Could not parse monomer SMILES for {record.poly_id}")

    frag_a = mol_to_fragment(mol_a, unit_tag=0.0)
    frag_b = mol_to_fragment(mol_b, unit_tag=1.0)
    node_features: list[list[float]] = []
    edge_pairs: list[list[int]] = []
    edge_features: list[list[float]] = []
    anchor_nodes: list[int] = []
    current_offset = 0

    for unit in sequence:
        frag_x, frag_edges, frag_edge_attr, anchor_idx = frag_a if unit == "A" else frag_b
        node_features.extend(frag_x)
        edge_pairs.extend([[src + current_offset, dst + current_offset] for src, dst in frag_edges])
        edge_features.extend(frag_edge_attr)
        anchor_nodes.append(anchor_idx + current_offset)
        current_offset += len(frag_x)

    virtual_edge = bond_features(None, is_virtual=True)
    for left_anchor, right_anchor in zip(anchor_nodes, anchor_nodes[1:]):
        edge_pairs.append([left_anchor, right_anchor])
        edge_pairs.append([right_anchor, left_anchor])
        edge_features.append(virtual_edge)
        edge_features.append(virtual_edge)

    global_features = torch.tensor([[record.frac_a, record.frac_b, abs(record.frac_a - record.frac_b)]], dtype=torch.float)
    target_value = record.ea if target_property == "EA" else record.ip
    label = 1 if target_value > threshold else 0
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edge_pairs, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_features, dtype=torch.float),
        y=torch.tensor([label], dtype=torch.long),
        u=global_features,
    )
    data.poly_id = record.poly_id
    data.poly_type_name = record.poly_type
    data.comp = record.comp
    data.target_value = target_value
    return data


def split_grouped(
    graphs: list[Data],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[Data], list[Data], list[Data]]:
    groups: dict[str, list[Data]] = {}
    for graph in graphs:
        groups.setdefault(graph.poly_id, []).append(graph)
    group_keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)
    total_groups = len(group_keys)
    train_cutoff = int(total_groups * train_ratio)
    val_cutoff = int(total_groups * (train_ratio + val_ratio))
    train_keys = set(group_keys[:train_cutoff])
    val_keys = set(group_keys[train_cutoff:val_cutoff])
    test_keys = set(group_keys[val_cutoff:])
    train_graphs = [graph for key in train_keys for graph in groups[key]]
    val_graphs = [graph for key in val_keys for graph in groups[key]]
    test_graphs = [graph for key in test_keys for graph in groups[key]]
    return train_graphs, val_graphs, test_graphs


def load_polymer_graphs(csv_path: Path, target_property: str, max_samples: int | None = None) -> list[Data]:
    records = load_polymer_records(csv_path, max_samples=max_samples)
    values = [record.ea if target_property == "EA" else record.ip for record in records]
    threshold = float(np.median(values))
    return [build_polymer_graph(record, target_property, threshold) for record in records]


def build_binary_graphs(dataset_csv: Path, smiles_column: str, target_column: str, id_column: str) -> tuple[list[Data], int]:
    dataframe = pd.read_csv(dataset_csv)
    required = {smiles_column, target_column, id_column}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {dataset_csv}: {missing}")

    graphs: list[Data] = []
    invalid_rows = 0
    empty_bond_dim = len(bond_features(None, is_virtual=False))
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

        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous() if edge_pairs else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else torch.empty((0, empty_bond_dim), dtype=torch.float)
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        global_features = torch.tensor(
            [[num_atoms / 100.0, num_bonds / 100.0, aromatic_atoms / max(num_atoms, 1)]],
            dtype=torch.float,
        )
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([int(target)], dtype=torch.long),
            u=global_features,
        )
        data.row_id = row_id
        data.smiles = smiles
        graphs.append(data)

    if not graphs:
        raise RuntimeError("No quedaron moleculas validas para el GNN binario.")
    return graphs, invalid_rows


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.u)
        loss = F.cross_entropy(logits, batch.y)
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


def run_vipea(args: argparse.Namespace) -> None:
    csv_path = args.csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    target_property = args.target_property
    property_stem = target_property.lower()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs = load_polymer_graphs(csv_path, target_property=target_property, max_samples=args.max_samples)
    if not graphs:
        raise RuntimeError("No se pudieron cargar grafos para el baseline GNN.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = graphs[0]
    overall_start = time.time()
    rows: list[dict[str, object]] = []
    confusion_matrices: list[list[list[int]]] = []
    best_row = None
    worst_row = None
    best_test_auroc = float("-inf")
    worst_test_auroc = float("inf")

    print(f"Loaded {len(graphs)} graphs")
    print(f"Target property: {target_property} (binarizada por mediana)")
    print(f"Iteraciones: {args.iterations}")
    print(f"Epochs por iteracion: {args.epochs}")

    for iteration in range(args.iterations):
        iteration_seed = args.random_state + iteration
        set_seed(iteration_seed)
        train_graphs, val_graphs, test_graphs = split_grouped(graphs, seed=iteration_seed)
        if not train_graphs or not val_graphs or not test_graphs:
            raise RuntimeError("Split failed: one of train/val/test is empty")

        train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
        test_loader = DataLoader(test_graphs, batch_size=args.batch_size)
        model = build_model(sample, args, num_classes=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_state = None
        best_val = -1.0
        start_time = time.time()
        for _epoch in range(1, args.epochs + 1):
            train_one_epoch(model, train_loader, optimizer, device)
            val_metrics = binary_metrics_from_scores(*collect_scores(model, val_loader, device))
            if val_metrics["accuracy"] > best_val:
                best_val = val_metrics["accuracy"]
                best_state = {key: value.cpu() for key, value in model.state_dict().items()}

        if best_state is None:
            raise RuntimeError("Training did not produce a checkpoint")

        model.load_state_dict(best_state)
        test_metrics = binary_metrics_from_scores(*collect_scores(model, test_loader, device))
        elapsed_seconds = float(time.time() - start_time)
        confusion_matrices.append(test_metrics["confusion_matrix"])
        row = {
            "Iteration": iteration + 1,
            "Random_State": iteration_seed,
            "Train_Size": len(train_graphs),
            "Val_Size": len(val_graphs),
            "Test_Size": len(test_graphs),
            "Best_Val_Accuracy": float(best_val),
            "Accuracy": test_metrics["accuracy"],
            "AUROC": test_metrics["roc_auc"],
            "Balanced_Accuracy": test_metrics["balanced_accuracy"],
            "F1": test_metrics["f1"],
            "Precision": test_metrics["precision"],
            "Recall": test_metrics["recall"],
            "Confusion_Matrix": json.dumps(test_metrics["confusion_matrix"], ensure_ascii=False),
            "Elapsed_Seconds": elapsed_seconds,
        }
        rows.append(row)

        current_auroc = float(test_metrics["roc_auc"])
        if current_auroc > best_test_auroc:
            best_test_auroc = current_auroc
            best_row = row
        if current_auroc < worst_test_auroc:
            worst_test_auroc = current_auroc
            worst_row = row

    metrics = {
        "pipeline": f"polymer_gnn_vipea_{property_stem}",
        "dataset_path": str(csv_path),
        "target_property": target_property,
        "iterations": args.iterations,
        "epochs": args.epochs,
        "average_metrics": {
            "Accuracy": float(np.mean([row["Accuracy"] for row in rows])),
            "AUROC": float(np.mean([row["AUROC"] for row in rows])),
            "Balanced_Accuracy": float(np.mean([row["Balanced_Accuracy"] for row in rows])),
            "F1": float(np.mean([row["F1"] for row in rows])),
            "Precision": float(np.mean([row["Precision"] for row in rows])),
            "Recall": float(np.mean([row["Recall"] for row in rows])),
            "Confusion_Matrix": average_confusion_matrix(confusion_matrices),
            "Elapsed_Seconds": float(np.mean([row["Elapsed_Seconds"] for row in rows])),
        },
        "best_iteration": best_row,
        "worst_iteration": worst_row,
        "timing_summary": {"total_wall_clock_seconds": float(time.time() - overall_start)},
    }
    write_json(output_dir / f"vipea_{property_stem}_polymer_gnn_metrics.json", metrics)
    write_csv(output_dir / f"vipea_{property_stem}_polymer_gnn_detailed_results.csv", rows)


def run_binary(args: argparse.Namespace) -> None:
    dataset_csv = args.dataset_csv
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_csv}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs, invalid_rows = build_binary_graphs(dataset_csv, args.smiles_column, args.target_column, args.id_column)
    process = psutil.Process() if psutil is not None else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = graphs[0]
    dataset_stem = dataset_csv.stem.lower()
    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    confusion_matrices: list[list[list[int]]] = []
    best_iteration_payload = None
    best_auroc = float("-inf")

    print(f"Loaded {len(graphs)} valid graphs from {dataset_csv}")
    print(f"Invalid rows skipped: {invalid_rows}")
    print(f"Iteraciones: {args.iterations}")
    print(f"Epochs maximos por iteracion: {args.epochs}")

    for iteration in range(args.iterations):
        iteration_seed = args.random_state + iteration
        set_seed(iteration_seed)
        cpu_time_start = get_process_cpu_time_seconds(process)
        start_time = time.time()

        labels = np.asarray([int(graph.y.item()) for graph in graphs], dtype=np.int64)
        indices = np.arange(len(graphs))
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

        best_state = None
        best_val_accuracy = float("-inf")
        epochs_without_improvement = 0
        for epoch in range(1, args.epochs + 1):
            train_one_epoch(model, train_loader, optimizer, device)
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

        model.load_state_dict(best_state)
        y_true, y_score, prediction_batch_rows = collect_binary_predictions(model, test_loader, device)
        test_metrics = binary_metrics_from_scores(y_true, y_score)
        elapsed_seconds = float(time.time() - start_time)
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
            "Training_Seconds": None,
            "Testing_Seconds": None,
        }
        rows.append(row)
        confusion_matrices.append(test_metrics["confusion_matrix"])

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

    metrics = {
        "pipeline": f"polymer_gnn_{dataset_stem}",
        "dataset_path": str(dataset_csv),
        "total_valid_graphs": len(graphs),
        "invalid_rows_skipped": invalid_rows,
        "iterations": args.iterations,
        "average_metrics": {
            "Accuracy": float(np.mean([row["Accuracy"] for row in rows])),
            "AUROC": float(np.mean([row["AUROC"] for row in rows])),
            "Balanced_Accuracy": float(np.mean([row["Balanced_Accuracy"] for row in rows])),
            "F1_Score": float(np.mean([row["F1_Score"] for row in rows])),
            "Precision": float(np.mean([row["Precision"] for row in rows])),
            "Recall": float(np.mean([row["Recall"] for row in rows])),
            "Confusion_Matrix": average_confusion_matrix(confusion_matrices),
        },
        "best_iteration_by_auroc": best_iteration_payload,
    }
    write_json(output_dir / f"{dataset_stem}_polymer_gnn_metrics.json", metrics)
    write_csv(output_dir / f"{dataset_stem}_polymer_gnn_detailed_results.csv", rows)
    write_csv(output_dir / f"{dataset_stem}_polymer_gnn_best_iteration_predictions.csv", prediction_rows)


def main() -> None:
    args = parse_args()
    if args.task == "vipea":
        run_vipea(args)
        return
    run_binary(args)
