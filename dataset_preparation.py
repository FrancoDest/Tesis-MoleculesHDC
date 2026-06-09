from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from catalog_config import REPO_ROOT, make_run_id, normalize_name, render_template


PTC_FM_NODE_LABEL_TO_SYMBOL = {
    0: "In",
    1: "P",
    2: "C",
    3: "O",
    4: "N",
    5: "Cl",
    6: "S",
    7: "Br",
    8: "Na",
    9: "F",
    10: "As",
    11: "K",
    12: "Cu",
    13: "I",
    14: "Ba",
    15: "Sn",
    16: "Pb",
    17: "Ca",
}

PTC_FM_EDGE_LABEL_TO_BOND = {
    0: "TRIPLE",
    1: "SINGLE",
    2: "DOUBLE",
    3: "AROMATIC",
}


def prepare_dataset_context(
    dataset_name: str,
    dataset_config: dict[str, Any],
    outputs_dir: Path,
) -> dict[str, str]:
    adapter = dataset_config.get("adapter")
    if not adapter:
        return {
            key: str(value)
            for key, value in dataset_config.items()
            if not isinstance(value, dict)
        }

    if adapter == "mutag_raw_pair_to_csv":
        return _prepare_mutag_csv(dataset_name, dataset_config, outputs_dir)
    if adapter == "deepchem_nci_to_csv":
        return _prepare_deepchem_nci_csv(dataset_name, dataset_config, outputs_dir)
    if adapter == "ptc_fm_raw_to_csv":
        return _prepare_ptc_fm_csv(dataset_name, dataset_config, outputs_dir)

    raise ValueError(f"Adapter de dataset no soportado para '{dataset_name}': {adapter}")


def resolve_execution_context(
    method_config: dict[str, Any],
    dataset_name: str,
    dataset_config: dict[str, Any],
    results_dir: Path,
) -> dict[str, Any]:
    method_name = normalize_name(method_config["name"])
    run_id = make_run_id()
    method_root = results_dir / method_name
    run_dir = method_root / dataset_name / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = run_dir / "_outputs"
    isolation_dir = run_dir / "_isolation"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    isolation_dir.mkdir(parents=True, exist_ok=True)

    workspace_root = REPO_ROOT
    workspace_copy_from = method_config.get("workspace_copy_from")
    if workspace_copy_from:
        source = (REPO_ROOT / workspace_copy_from).resolve()
        workspace_root = run_dir / "_workspace"
        shutil.copytree(source, workspace_root, dirs_exist_ok=True)
        cwd = (workspace_root / method_config.get("cwd_in_workspace", ".")).resolve()
    else:
        cwd = (REPO_ROOT / method_config.get("cwd", ".")).resolve()

    template_context = {
        "repo_root": str(REPO_ROOT),
        "workspace_root": str(workspace_root),
        "run_dir": str(run_dir),
        "run_output_dir": str(outputs_dir),
        "method_name": method_name,
        "dataset_name": dataset_name,
    }
    dataset_context = prepare_dataset_context(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        outputs_dir=outputs_dir,
    )
    template_context.update(dataset_context)

    summary_patterns = render_template(method_config.get("summary_patterns", []), template_context)

    return {
        "method_name": method_name,
        "dataset_name": dataset_name,
        "run_id": run_id,
        "run_dir": run_dir,
        "run_log_path": run_dir / "run.log",
        "cwd": cwd,
        "summary_patterns": summary_patterns,
        "isolation_dir": isolation_dir,
        "outputs_dir": outputs_dir,
        "workspace_root": workspace_root,
        "template_context": template_context,
    }


def _prepare_mutag_csv(
    dataset_name: str,
    dataset_config: dict[str, Any],
    outputs_dir: Path,
) -> dict[str, str]:
    smiles_path = (REPO_ROOT / str(dataset_config["source_smiles_path"])).resolve()
    labels_path = (REPO_ROOT / str(dataset_config["source_labels_path"])).resolve()
    if not smiles_path.exists():
        raise FileNotFoundError(f"No existe archivo de smiles para {dataset_name}: {smiles_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"No existe archivo de labels para {dataset_name}: {labels_path}")

    smiles_lines = [line.strip() for line in smiles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    label_lines = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(smiles_lines) != len(label_lines):
        raise ValueError(
            f"MUTAG inconsistente: {len(smiles_lines)} smiles y {len(label_lines)} labels."
        )

    prepared_dir = outputs_dir / "prepared_dataset"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_csv = prepared_dir / "MUTAG.csv"

    with prepared_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "smiles", "label"])
        writer.writeheader()
        for index, (smiles_line, label_line) in enumerate(zip(smiles_lines, label_lines)):
            smiles = smiles_line.split()[0]
            label_value = int(label_line)
            writer.writerow(
                {
                    "id": str(index),
                    "smiles": smiles,
                    "label": 1 if label_value > 0 else 0,
                }
            )

    context = {
        key: str(value)
        for key, value in dataset_config.items()
        if not isinstance(value, dict)
    }
    context.update(
        {
            "dataset_path": "/run_outputs/prepared_dataset/MUTAG.csv",
            "embeddings_csv": "/run_outputs/artifacts/mutag_mol2vec_features.csv",
            "rdkit_valid_csv": "/run_outputs/prepared_dataset/MUTAG.csv",
            "labels_csv": "/run_outputs/prepared_dataset/MUTAG.csv",
            "prepared_dataset_csv": "/run_outputs/prepared_dataset/MUTAG.csv",
            "rf_dataset_path": "/run_outputs/prepared_dataset/MUTAG.csv",
            "molebert_input_csv": "/run_outputs/prepared_dataset/MUTAG.csv",
            "molehd_dataset_path": "/run_outputs/prepared_dataset/MUTAG.csv",
            "graphhd_dataset_path": "/run_outputs/prepared_dataset/MUTAG.csv",
            "polymer_gnn_dataset_path": "/run_outputs/prepared_dataset/MUTAG.csv",
        }
    )
    return context


def _prepare_deepchem_nci_csv(
    dataset_name: str,
    dataset_config: dict[str, Any],
    outputs_dir: Path,
) -> dict[str, str]:
    task_name = dataset_config.get("task_name")
    try:
        from deepchem.molnet import load_nci  # type: ignore
    except ImportError:
        try:
            from deepchem.molnet import load_nci1 as load_nci  # type: ignore
        except ImportError:
            _prepare_nci_csv_with_docker(outputs_dir, task_name=task_name)
            return _build_binary_dataset_context(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                prepared_csv=outputs_dir / "prepared_dataset" / "NCI1.csv",
                embeddings_stem="nci1",
            )

    prepared_csv = outputs_dir / "prepared_dataset" / "NCI1.csv"
    _write_nci_rows_to_csv(load_nci=load_nci, prepared_csv=prepared_csv, task_name=task_name)

    return _build_binary_dataset_context(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        prepared_csv=prepared_csv,
        embeddings_stem="nci1",
    )


def _prepare_ptc_fm_csv(
    dataset_name: str,
    dataset_config: dict[str, Any],
    outputs_dir: Path,
) -> dict[str, str]:
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - depende del runtime
        raise ImportError(
            "PTC_FM necesita RDKit para reconstruir los grafos a SMILES, "
            "pero RDKit no esta disponible en este entorno."
        ) from exc

    source_dir = (REPO_ROOT / str(dataset_config["source_dir"])).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta raw de PTC_FM: {source_dir}")

    graph_indicator = _read_int_lines(source_dir / "PTC_FM_graph_indicator.txt")
    node_labels = _read_int_lines(source_dir / "PTC_FM_node_labels.txt")
    graph_labels = _read_int_lines(source_dir / "PTC_FM_graph_labels.txt")
    edge_labels = _read_int_lines(source_dir / "PTC_FM_edge_labels.txt")
    adjacency_pairs = _read_edge_pairs(source_dir / "PTC_FM_A.txt")

    if len(graph_indicator) != len(node_labels):
        raise ValueError(
            "PTC_FM inconsistente: graph_indicator y node_labels no tienen el mismo largo."
        )
    if len(adjacency_pairs) != len(edge_labels):
        raise ValueError(
            "PTC_FM inconsistente: A.txt y edge_labels.txt no tienen el mismo largo."
        )

    nodes_by_graph: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for global_node_id, (graph_id, node_label) in enumerate(zip(graph_indicator, node_labels), start=1):
        nodes_by_graph[graph_id].append((global_node_id, node_label))

    edges_by_graph: dict[int, dict[tuple[int, int], int]] = defaultdict(dict)
    for (left, right), edge_label in zip(adjacency_pairs, edge_labels):
        graph_id = graph_indicator[left - 1]
        if graph_indicator[right - 1] != graph_id:
            raise ValueError(
                f"PTC_FM inconsistente: arista entre grafos distintos ({left}, {right})."
            )
        edge_key = tuple(sorted((left, right)))
        edges_by_graph[graph_id].setdefault(edge_key, edge_label)

    prepared_dir = outputs_dir / "prepared_dataset"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_csv = prepared_dir / "PTC_FM.csv"

    invalid_graphs = 0
    written_rows = 0
    with prepared_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "smiles", "label"])
        writer.writeheader()

        for graph_index, graph_label in enumerate(graph_labels, start=1):
            mol = Chem.RWMol()
            global_to_local: dict[int, int] = {}

            for global_node_id, node_label in nodes_by_graph.get(graph_index, []):
                symbol = PTC_FM_NODE_LABEL_TO_SYMBOL.get(node_label)
                if symbol is None:
                    raise ValueError(
                        f"PTC_FM tiene un node_label desconocido: {node_label} en grafo {graph_index}."
                    )
                atom_idx = mol.AddAtom(Chem.Atom(symbol))
                global_to_local[global_node_id] = atom_idx

            for (left, right), edge_label in edges_by_graph.get(graph_index, {}).items():
                bond_name = PTC_FM_EDGE_LABEL_TO_BOND.get(edge_label)
                if bond_name is None:
                    raise ValueError(
                        f"PTC_FM tiene un edge_label desconocido: {edge_label} en grafo {graph_index}."
                    )
                bond_type = getattr(Chem.rdchem.BondType, bond_name)
                mol.AddBond(global_to_local[left], global_to_local[right], bond_type)
                if edge_label == 3:
                    bond = mol.GetBondBetweenAtoms(global_to_local[left], global_to_local[right])
                    if bond is not None:
                        bond.SetIsAromatic(True)
                    mol.GetAtomWithIdx(global_to_local[left]).SetIsAromatic(True)
                    mol.GetAtomWithIdx(global_to_local[right]).SetIsAromatic(True)

            final_mol = mol.GetMol()
            try:
                Chem.SanitizeMol(final_mol)
                smiles = Chem.MolToSmiles(final_mol, canonical=True)
            except Exception:
                invalid_graphs += 1
                continue

            if not smiles:
                invalid_graphs += 1
                continue

            writer.writerow(
                {
                    "id": str(graph_index - 1),
                    "smiles": smiles,
                    "label": _normalize_binary_label(graph_label),
                }
            )
            written_rows += 1

    if written_rows == 0:
        raise ValueError(
            "No pude reconstruir ninguna molecula valida para PTC_FM con RDKit."
        )

    return _build_binary_dataset_context(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        prepared_csv=prepared_csv,
        embeddings_stem="ptc_fm",
    )


def _build_binary_dataset_context(
    *,
    dataset_name: str,
    dataset_config: dict[str, Any],
    prepared_csv: Path,
    embeddings_stem: str,
) -> dict[str, str]:
    del dataset_name
    context = {
        key: str(value)
        for key, value in dataset_config.items()
        if not isinstance(value, dict)
    }
    prepared_path = str(prepared_csv)
    container_prepared_path = f"/run_outputs/prepared_dataset/{prepared_csv.name}"
    context.update(
        {
            "dataset_path": prepared_path,
            "embeddings_csv": f"/run_outputs/artifacts/{embeddings_stem}_mol2vec_features.csv",
            "rdkit_valid_csv": container_prepared_path,
            "labels_csv": container_prepared_path,
            "prepared_dataset_csv": container_prepared_path,
            "rf_dataset_path": container_prepared_path,
            "molebert_input_csv": container_prepared_path,
            "molehd_dataset_path": container_prepared_path,
            "graphhd_dataset_path": container_prepared_path,
            "polymer_gnn_dataset_path": container_prepared_path,
        }
    )
    return context


def _prepare_nci_csv_with_docker(outputs_dir: Path, task_name: str | None) -> None:
    image_name = "final-nci-dataset-prep"
    dockerfile = REPO_ROOT / "final" / "nci_dataset_prep_docker" / "Dockerfile"
    build_context = (REPO_ROOT / "final").resolve()
    prepared_dir = outputs_dir / "prepared_dataset"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    inspect = subprocess.run(
        ["docker", "image", "inspect", image_name],
        capture_output=True,
        text=True,
    )
    if inspect.returncode != 0:
        subprocess.run(
            [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
                "-t",
                image_name,
                "-f",
                str(dockerfile.resolve()),
                str(build_context),
            ],
            check=True,
        )

    env = dict(os.environ)
    env.setdefault("DOCKER_DEFAULT_PLATFORM", "linux/amd64")
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-e",
            "PYTHONIOENCODING=utf-8",
            "-v",
            f"{prepared_dir.resolve()}:/run_outputs/prepared_dataset",
            image_name,
            "python",
            "/opt/nci_prep/prepare_nci_dataset.py",
            "--output-csv",
            "/run_outputs/prepared_dataset/NCI1.csv",
            *([] if not task_name else ["--task-name", task_name]),
        ],
        check=True,
        env=env,
    )


def _select_nci_task_index(tasks: list[str], task_name: str | None) -> int:
    if not tasks:
        raise ValueError("DeepChem no devolvio tareas para NCI/NCL1.")
    if task_name is None:
        return 0
    try:
        return tasks.index(task_name)
    except ValueError as exc:
        raise ValueError(
            f"La tarea pedida para NCI/NCL1 no existe: {task_name!r}. "
            f"Primeras tareas disponibles: {tasks[:5]!r}"
        ) from exc


def _write_nci_rows_to_csv(*, load_nci: Any, prepared_csv: Path, task_name: str | None) -> None:
    tasks, datasets, _transformers = load_nci(featurizer="Raw", splitter=None)
    if not datasets:
        raise ValueError("DeepChem no devolvio datasets para NCI1.")
    task_index = _select_nci_task_index(list(tasks), task_name)

    prepared_csv.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    written_rows = 0
    invalid_rows = 0
    observed_labels: set[int] = set()

    with prepared_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "smiles", "label"])
        writer.writeheader()

        for split_index, dataset in enumerate(datasets):
            ids = list(getattr(dataset, "ids", []))
            labels = getattr(dataset, "y", None)
            if labels is None:
                raise ValueError("DeepChem devolvio un dataset sin etiquetas en .y.")

            if len(ids) != len(labels):
                raise ValueError(
                    f"DeepChem devolvio ids y labels con distinto largo en split {split_index}: "
                    f"{len(ids)} vs {len(labels)}."
                )

            for row_index, (row_id, raw_label) in enumerate(zip(ids, labels)):
                smiles = str(row_id).strip()
                if not smiles:
                    invalid_rows += 1
                    continue

                flattened = _flatten_label_value(raw_label[task_index])
                if flattened is None:
                    invalid_rows += 1
                    continue

                if math.isnan(flattened):
                    invalid_rows += 1
                    continue
                normalized = _normalize_binary_label(flattened)
                observed_labels.add(normalized)
                unique_id = f"{split_index}_{row_index}"
                if unique_id in seen_ids:
                    continue
                seen_ids.add(unique_id)

                writer.writerow(
                    {
                        "id": unique_id,
                        "smiles": smiles,
                        "label": normalized,
                    }
                )
                written_rows += 1

    if written_rows == 0:
        raise ValueError(
            "No pude generar filas validas para NCI1 desde DeepChem. "
            f"Filas invalidas descartadas: {invalid_rows}"
        )
    if observed_labels != {0, 1}:
        raise ValueError(
            "NCI/NCL1 no termino teniendo las dos clases binarias esperadas "
            f"despues de normalizar labels: {sorted(observed_labels)!r}"
        )


def _read_int_lines(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"No existe archivo requerido: {path}")
    values: list[int] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        values.append(int(line))
    return values


def _read_edge_pairs(path: Path) -> list[tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"No existe archivo requerido: {path}")
    pairs: list[tuple[int, int]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        left, right = [piece.strip() for piece in line.split(",", maxsplit=1)]
        pairs.append((int(left), int(right)))
    return pairs


def _flatten_label_value(value: Any) -> float | None:
    current = value
    while isinstance(current, (list, tuple)) and current:
        current = current[0]

    shape = getattr(current, "shape", None)
    if shape is not None:
        size = getattr(current, "size", None)
        if size == 0:
            return None
        if hasattr(current, "reshape"):
            current = current.reshape(-1)[0]

    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def _normalize_binary_label(value: float | int) -> int:
    numeric_value = float(value)
    return 1 if numeric_value > 0 else 0
