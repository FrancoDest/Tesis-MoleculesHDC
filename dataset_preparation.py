from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from catalog_config import REPO_ROOT, make_run_id, normalize_name, render_template


def prepare_dataset_context(
    dataset_name: str,
    dataset_config: dict[str, Any],
    outputs_dir: Path,
) -> dict[str, str]:
    adapter = dataset_config.get("adapter")
    if not adapter:
        context = {
            key: str(value)
            for key, value in dataset_config.items()
            if not isinstance(value, dict)
        }
        context.setdefault("graph_dataset_path", "")
        return _with_optional_feature_defaults(context)

    if adapter == "tu_dataset_graph":
        context = _prepare_tu_graph_npz(dataset_name, dataset_config, outputs_dir)
    elif adapter == "refractive_index_binarize":
        context = _prepare_refractive_index_csv(dataset_name, dataset_config, outputs_dir)
    elif adapter == "glass_transition_mol_to_csv":
        context = _prepare_glass_transition_csv(dataset_name, dataset_config, outputs_dir)
    else:
        raise ValueError(f"Adapter de dataset no soportado para '{dataset_name}': {adapter}")

    return _with_optional_feature_defaults(context)


#  n_jobs es por (metodo, dataset), no solo por dataset: distintos pipelines
# tienen distinta heuristica automatica y distinto perfil de memoria sobre el
# mismo dataset (ej. mole_bert_hdc con JL a 10048d es mucho mas sensible a
# RAM que mole_bert_rf, que no proyecta). Cada metodo referencia su propia
# clave {n_jobs_<metodo>} en su run_args; se lista aca para poder rellenar
# "0" (= heuristica automatica del pipeline) por defecto en todos los
# datasets que no la pisen explicitamente.
_N_JOBS_OVERRIDE_KEYS = (
    "n_jobs_mole_bert_hdc",
    "n_jobs_mol2vec_jl",
)


def _with_optional_feature_defaults(context: dict[str, str]) -> dict[str, str]:
    """Los run_args de los pipelines SMILES referencian
    {frac_a_column}/{frac_b_column}/{group_column} incondicionalmente (el
    mismo run_args sirve para todos los datasets del metodo). Datasets que no
    definen estas columnas (bbbp, hiv, etc.) necesitan que el placeholder
    resuelva igual a "" en vez de tirar KeyError en el .format(); los
    pipelines tratan "" como "no pasado" (falsy) y no suman la feature."""
    context.setdefault("frac_a_column", "")
    context.setdefault("frac_b_column", "")
    context.setdefault("group_column", "")
    for key in _N_JOBS_OVERRIDE_KEYS:
        context.setdefault(key, "0")
    return context


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


def _prepare_tu_graph_npz(
    dataset_name: str,
    dataset_config: dict[str, Any],
    outputs_dir: Path,
) -> dict[str, str]:
    """Datasets grafo-originales (benchmarks TU Dataset: MUTAG, PTC_FM, NCI1):
    nunca pasan por SMILES/RDKit. Se leen los archivos crudos TU (`_A.txt`,
    `_graph_indicator.txt`, `_graph_labels.txt`, `_node_labels.txt`, y
    `_edge_labels.txt` si el dataset lo trae) y se repaquetan tal cual -- sin
    reconstruir ninguna molecula, sin tablas de mapeo atomo/enlace, sin
    perdida -- en un unico .npz que graphHD/polymer_gnn consumen directo."""
    source_dir = (REPO_ROOT / str(dataset_config["source_dir"])).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta raw de {dataset_name}: {source_dir}")

    graph_label_files = sorted(source_dir.glob("*_graph_labels.txt"))
    if not graph_label_files:
        raise FileNotFoundError(f"No encontre *_graph_labels.txt en {source_dir}")
    prefix = graph_label_files[0].name[: -len("_graph_labels.txt")]

    def _tu_path(suffix: str) -> Path:
        return source_dir / f"{prefix}_{suffix}"

    graph_indicator = _read_int_lines(_tu_path("graph_indicator.txt"))
    node_labels = _read_int_lines(_tu_path("node_labels.txt"))
    graph_labels = _read_int_lines(_tu_path("graph_labels.txt"))
    adjacency_pairs = _read_edge_pairs(_tu_path("A.txt"))

    edge_labels_path = _tu_path("edge_labels.txt")
    edge_labels = _read_int_lines(edge_labels_path) if edge_labels_path.exists() else None

    if len(graph_indicator) != len(node_labels):
        raise ValueError(
            f"{dataset_name} inconsistente: graph_indicator y node_labels no tienen el mismo largo."
        )
    if edge_labels is not None and len(edge_labels) != len(adjacency_pairs):
        raise ValueError(
            f"{dataset_name} inconsistente: A.txt y edge_labels.txt no tienen el mismo largo."
        )

    # TU Dataset es 1-indexado (nodos y grafos); se pasa todo a 0-indexado aca
    # para que quien consuma el .npz no tenga que pensar en el offset.
    graph_indicator_zero = np.asarray(graph_indicator, dtype=np.int64) - 1
    node_labels_array = np.asarray(node_labels, dtype=np.int64)
    graph_labels_array = np.asarray(
        [_normalize_binary_label(value) for value in graph_labels], dtype=np.int64
    )
    if adjacency_pairs:
        edge_index = np.asarray(
            [[left - 1, right - 1] for left, right in adjacency_pairs], dtype=np.int64
        ).T
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)

    prepared_dir = outputs_dir / "prepared_dataset"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_npz = prepared_dir / f"{dataset_name}_graph.npz"

    savez_kwargs: dict[str, np.ndarray] = {
        "edge_index": edge_index,
        "node_labels": node_labels_array,
        "graph_indicator": graph_indicator_zero,
        "graph_labels": graph_labels_array,
    }
    if edge_labels is not None:
        savez_kwargs["edge_labels"] = np.asarray(edge_labels, dtype=np.int64)
    np.savez(prepared_npz, **savez_kwargs)

    context = {
        key: str(value)
        for key, value in dataset_config.items()
        if not isinstance(value, dict)
    }
    context.update(
        {
            # Ningun pipeline SMILES corre sobre datasets grafo-originales:
            # estas keys quedan vacias a proposito.
            "dataset_path": "",
            "embeddings_csv": "",
            "rdkit_valid_csv": "",
            "labels_csv": "",
            "prepared_dataset_csv": "",
            "rf_dataset_path": "",
            "molebert_input_csv": "",
            "molehd_dataset_path": "",
            "graphhd_dataset_path": "",
            "polymer_gnn_dataset_path": "",
            "graph_dataset_path": f"/run_outputs/prepared_dataset/{prepared_npz.name}",
        }
    )
    return context


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _prepare_refractive_index_csv(
    dataset_name: str,
    dataset_config: dict[str, Any],
    outputs_dir: Path,
) -> dict[str, str]:
    """El Refractive Index es un valor continuo (regresion); se binariza por
    la mediana (0 = por debajo, 1 = igual o por encima) para poder usar los
    pipelines de clasificacion binaria tal cual estan, sin agregarles soporte
    de regresion. El CSV fuente trae una fila de titulo suelta antes del
    encabezado real, asi que se busca la fila de encabezado en vez de asumir
    que es la primera."""
    source_csv_path = (REPO_ROOT / str(dataset_config["source_csv"])).resolve()
    if not source_csv_path.exists():
        raise FileNotFoundError(f"No existe archivo fuente para {dataset_name}: {source_csv_path}")

    with source_csv_path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        raw_rows = list(csv.reader(handle))

    header_index = None
    for index, row in enumerate(raw_rows):
        if "Refractive Index" in row and "SMILES" in row:
            header_index = index
            break
    if header_index is None:
        raise ValueError(f"No encontre el encabezado esperado (Refractive Index, SMILES) en {source_csv_path}")

    header = raw_rows[header_index]
    parsed_rows: list[dict[str, object]] = []
    for raw_row in raw_rows[header_index + 1 :]:
        if not any(cell.strip() for cell in raw_row):
            continue
        record = dict(zip(header, raw_row))
        smiles = (record.get("SMILES") or "").strip()
        value_text = (record.get("Refractive Index") or "").strip()
        if not smiles or not value_text:
            continue
        try:
            value = float(value_text)
        except ValueError:
            continue
        parsed_rows.append({"smiles": smiles, "value": value})

    if not parsed_rows:
        raise ValueError(f"No pude leer filas validas de {source_csv_path}")

    median = _median([row["value"] for row in parsed_rows])

    prepared_dir = outputs_dir / "prepared_dataset"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_csv = prepared_dir / f"{dataset_name}.csv"
    with prepared_csv.open("w", newline="", encoding="utf-8") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=["id", "smiles", "label", "refractive_index"])
        writer.writeheader()
        for index, row in enumerate(parsed_rows):
            writer.writerow(
                {
                    "id": str(index),
                    "smiles": row["smiles"],
                    "label": 1 if row["value"] >= median else 0,
                    "refractive_index": row["value"],
                }
            )

    return _build_binary_dataset_context(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        prepared_csv=prepared_csv,
        embeddings_stem=dataset_name,
    )


# Fuente: Palomba, Vazquez & Diaz (2012), "Novel descriptors from main and
# side chains of high-molecular-weight polymers applied to prediction of
# glass transition temperatures", J. Mol. Graph. Model. 38, 137-147, Table 1.
# El repo solo tenia los 88 archivos .mol (trimeros) y el PDF del paper, sin
# ningun CSV con los valores experimentales Tg/M -- se transcribieron a mano
# desde la Tabla 1 del PDF (columna "Exp.", en K*mol/g). El indice coincide
# con el prefijo numerico de cada archivo .mol (ej. "001 trimer poly(ethylene).mol" -> 1).
_GLASS_TRANSITION_TG_M_BY_INDEX: dict[int, float] = {
    1: 6.96, 2: 4.07, 3: 2.62, 4: 3.63, 5: 3.30, 6: 5.26, 7: 3.27, 8: 2.51,
    9: 1.98, 10: 8.14, 11: 5.57, 12: 7.13, 13: 3.50, 14: 3.59, 15: 2.84,
    16: 2.63, 17: 2.82, 18: 3.47, 19: 3.17, 20: 3.17, 21: 3.11, 22: 5.55,
    23: 3.14, 24: 3.53, 25: 2.46, 26: 1.71, 27: 1.63, 28: 3.55, 29: 2.64,
    30: 3.64, 31: 3.47, 32: 3.78, 33: 2.84, 34: 2.55, 35: 2.73, 36: 2.47,
    37: 2.68, 38: 2.43, 39: 3.22, 40: 7.27, 41: 4.68, 42: 3.36, 43: 2.64,
    44: 1.80, 45: 1.24, 46: 1.07, 47: 1.59, 48: 2.04, 49: 1.82, 50: 1.33,
    51: 1.13, 52: 1.28, 53: 1.25, 54: 1.09, 55: 1.38, 56: 2.24, 57: 2.21,
    58: 2.01, 59: 2.56, 60: 2.53, 61: 2.15, 62: 2.76, 63: 1.87, 64: 2.51,
    65: 2.28, 66: 4.61, 67: 2.04, 68: 2.32, 69: 1.69, 70: 1.81, 71: 2.64,
    72: 2.09, 73: 1.58, 74: 3.14, 75: 1.74, 76: 2.06, 77: 2.03, 78: 3.13,
    79: 3.64, 80: 2.53, 81: 2.39, 82: 3.03, 83: 2.82, 84: 2.32, 85: 2.14,
    86: 2.81, 87: 3.11, 88: 3.78,
}


def _prepare_glass_transition_csv(
    dataset_name: str,
    dataset_config: dict[str, Any],
    outputs_dir: Path,
) -> dict[str, str]:
    """No hay CSV fuente para este dataset: solo 88 archivos .mol (trimeros)
    y el paper en PDF. Convierte cada .mol a SMILES con RDKit, le pega el
    Tg/M experimental (ver _GLASS_TRANSITION_TG_M_BY_INDEX) por el indice
    numerico del nombre de archivo, y binariza por la mediana."""
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - depende del runtime
        raise ImportError(
            "El dataset de glass transition temperature ratio necesita RDKit "
            "para convertir los archivos .mol a SMILES, pero RDKit no esta "
            "disponible en este entorno."
        ) from exc

    source_dir = (REPO_ROOT / str(dataset_config["source_mol_dir"])).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de archivos .mol: {source_dir}")

    mol_files = sorted(source_dir.glob("*.mol"))
    if not mol_files:
        raise FileNotFoundError(f"No encontre archivos .mol en {source_dir}")

    parsed_rows: list[dict[str, object]] = []
    skipped = 0
    for mol_path in mol_files:
        match = re.match(r"\s*(\d+)", mol_path.stem)
        if not match:
            skipped += 1
            continue
        index = int(match.group(1))
        tg_m = _GLASS_TRANSITION_TG_M_BY_INDEX.get(index)
        if tg_m is None:
            skipped += 1
            continue

        mol = Chem.MolFromMolFile(str(mol_path))
        if mol is None:
            skipped += 1
            continue
        try:
            smiles = Chem.MolToSmiles(mol)
        except Exception:
            skipped += 1
            continue
        if not smiles:
            skipped += 1
            continue

        parsed_rows.append({"index": index, "smiles": smiles, "value": tg_m, "name": mol_path.stem})

    if not parsed_rows:
        raise ValueError(f"No pude convertir ningun archivo .mol a SMILES en {source_dir}")

    median = _median([row["value"] for row in parsed_rows])
    parsed_rows.sort(key=lambda row: row["index"])

    prepared_dir = outputs_dir / "prepared_dataset"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_csv = prepared_dir / f"{dataset_name}.csv"
    with prepared_csv.open("w", newline="", encoding="utf-8") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=["id", "smiles", "label", "tg_m_ratio", "polymer_name"])
        writer.writeheader()
        for row in parsed_rows:
            writer.writerow(
                {
                    "id": str(row["index"]),
                    "smiles": row["smiles"],
                    "label": 1 if row["value"] >= median else 0,
                    "tg_m_ratio": row["value"],
                    "polymer_name": row["name"],
                }
            )

    return _build_binary_dataset_context(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        prepared_csv=prepared_csv,
        embeddings_stem=dataset_name,
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
            "graph_dataset_path": "",
        }
    )
    return context


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


def _normalize_binary_label(value: float | int) -> int:
    numeric_value = float(value)
    return 1 if numeric_value > 0 else 0
