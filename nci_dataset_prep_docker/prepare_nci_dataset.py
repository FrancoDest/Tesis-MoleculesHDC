from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara NCI1/NCL1 desde DeepChem MolNet y lo exporta a CSV binario."
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Ruta de salida para el CSV preparado.",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="Nombre exacto de la tarea de NCI60 a proyectar como binaria de una sola etiqueta.",
    )
    return parser.parse_args()


def flatten_label_value(value: Any) -> float | None:
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


def normalize_binary_label(value: float | int) -> int:
    return 1 if float(value) > 0 else 0


def select_task_index(tasks: list[str], task_name: str | None) -> int:
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


def main() -> None:
    args = parse_args()

    try:
        from deepchem.molnet import load_nci  # type: ignore
    except ImportError:
        from deepchem.molnet import load_nci1 as load_nci  # type: ignore

    tasks, datasets, _transformers = load_nci(featurizer="Raw", splitter=None)
    if not datasets:
        raise ValueError("DeepChem no devolvio datasets para NCI1.")
    task_index = select_task_index(list(tasks), args.task_name)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    observed_labels: set[int] = set()
    written_rows = 0

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
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
                    continue

                flattened = flatten_label_value(raw_label[task_index])
                if flattened is None:
                    continue
                if math.isnan(flattened):
                    continue

                normalized = normalize_binary_label(flattened)
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
        raise ValueError("No pude generar filas validas para NCI1 desde DeepChem.")
    if observed_labels != {0, 1}:
        raise ValueError(
            "NCI/NCL1 no termino teniendo las dos clases binarias esperadas "
            f"despues de normalizar labels: {sorted(observed_labels)!r}"
        )


if __name__ == "__main__":
    main()
