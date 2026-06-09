from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from catalog_config import (
    DEFAULT_CATALOG,
    DEFAULT_RESULTS_DIR,
    ensure_parent,
    join_command,
    list_options,
    load_catalog,
    prepare_isolated_env,
    resolve_python,
    select_methods,
)
from dataset_preparation import resolve_execution_context
from docker_runtime import build_docker_image, build_docker_run_command

DEFAULT_DATASET = "bbbp"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta benchmarks de forma aislada, eligiendo dataset y metodo "
            "desde un catalogo entendible."
        )
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help="JSON con datasets y metodos disponibles.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Carpeta raiz para corridas, logs y CSVs.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help="Interpreter Python local. Solo se usa para herramientas auxiliares del runner.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Dataset a usar segun el catalogo. Default: {DEFAULT_DATASET}.",
    )
    parser.add_argument(
        "--method",
        nargs="*",
        default=None,
        help="Uno o varios metodos del catalogo. Si no se pasa, usa todos los compatibles con el dataset.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lista datasets y metodos disponibles sin ejecutar nada.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra build y run resueltos sin ejecutarlos.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Fuerza reconstruir la imagen Docker del metodo antes de correr.",
    )
    return parser.parse_args()


def filter_methods_inside_final(methods: list[dict]) -> tuple[list[dict], list[tuple[str, str]]]:
    available_methods = []
    skipped_methods = []
    for method_config in methods:
        build_context = method_config.get("docker", {}).get("build_context")
        workspace_copy_from = method_config.get("workspace_copy_from")
        candidate = build_context or workspace_copy_from
        if candidate is None:
            available_methods.append(method_config)
            continue

        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = PROJECT_ROOT / candidate

        if candidate_path.exists() and candidate_path.is_relative_to(FINAL_ROOT):
            available_methods.append(method_config)
        else:
            skipped_methods.append((method_config["name"], str(candidate_path)))

    return available_methods, skipped_methods


def flatten_dict(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            next_prefix = f"{prefix}_{key}" if prefix else str(key)
            flatten_dict(next_prefix, nested_value, output)
        return
    if isinstance(value, list):
        output[prefix] = json.dumps(make_json_safe(value), ensure_ascii=False)
        return
    output[prefix] = make_json_safe(value)


def make_json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, dict):
        return {key: make_json_safe(nested_value) for key, nested_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def summarize_json_metrics(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    row: dict[str, Any] = {"summary_source": str(path), "summary_format": "json"}
    interesting_keys = (
        "pipeline",
        "dataset_path",
        "source_embeddings_csv",
        "source_labels_csv",
        "iterations",
        "random_state",
        "random_state_base",
        "test_size",
        "test_fraction",
        "train_size",
        "eval_size",
        "train_size_first_iteration",
        "test_size_first_iteration",
        "total_rows_original",
        "total_rows_valid_rdkit",
        "total_rows_with_embeddings_and_labels",
        "n_estimators",
        "jl_dim",
    )
    for key in interesting_keys:
        if key in payload:
            row[key] = payload[key]

    for payload_key, prefix in {
        "average_metrics": "average",
        "best_run": "best",
        "worst_run": "worst",
        "best_iteration_by_auroc": "best",
        "worst_iteration_by_auroc": "worst",
        "average_resources": "resources",
        "timing_summary": "timing",
    }.items():
        if payload_key in payload:
            flatten_dict(prefix, payload[payload_key], row)

    if "average_confusion_matrix" in payload:
        row["average_confusion_matrix"] = json.dumps(make_json_safe(payload["average_confusion_matrix"]), ensure_ascii=False)

    return [row]


def summarize_pickle_metrics(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    row: dict[str, Any] = {"summary_source": str(path), "summary_format": "pickle"}
    for key in ("target", "config", "binarization", "threshold"):
        if key in payload:
            row[key] = payload[key]

    list_mappings = {
        "accuracy_list": "average_accuracy",
        "auroc_list": "average_auroc",
        "bacc_list": "average_bacc",
        "f1_list": "average_f1",
        "precision_list": "average_precision",
        "recall_list": "average_recall",
        "cpu_time_seconds_list": "average_cpu_total_time_seconds",
        "memory_percent_list": "average_memory_percent",
        "memory_mb_list": "average_memory_mb",
        "elapsed_seconds_list": "average_elapsed_seconds",
        "training_seconds_list": "average_training_seconds",
        "testing_seconds_list": "average_testing_seconds",
    }
    for key, target_key in list_mappings.items():
        values = payload.get(key)
        if isinstance(values, list) and values:
            row[target_key] = sum(values) / len(values)

    auroc_values = payload.get("auroc_list")
    if isinstance(auroc_values, list) and auroc_values:
        best_idx = max(range(len(auroc_values)), key=auroc_values.__getitem__)
        worst_idx = min(range(len(auroc_values)), key=auroc_values.__getitem__)
        row["best_auroc"] = auroc_values[best_idx]
        row["worst_auroc"] = auroc_values[worst_idx]
        if "accuracy_list" in payload:
            row["best_accuracy"] = payload["accuracy_list"][best_idx]
            row["worst_accuracy"] = payload["accuracy_list"][worst_idx]
        if "confusion_matrices" in payload:
            row["best_confusion_matrix"] = json.dumps(make_json_safe(payload["confusion_matrices"][best_idx]), ensure_ascii=False)
            row["worst_confusion_matrix"] = json.dumps(make_json_safe(payload["confusion_matrices"][worst_idx]), ensure_ascii=False)
        if "random_states" in payload:
            row["best_random_state"] = payload["random_states"][best_idx]
            row["worst_random_state"] = payload["random_states"][worst_idx]

    if "average_confusion_matrix" in payload:
        row["average_confusion_matrix"] = json.dumps(make_json_safe(payload["average_confusion_matrix"]), ensure_ascii=False)
    if "fixed_confusion_matrix" in payload and payload["fixed_confusion_matrix"] is not None:
        row["fixed_confusion_matrix"] = json.dumps(make_json_safe(payload["fixed_confusion_matrix"]), ensure_ascii=False)
    for balance_key in ("global_class_balance", "fixed_train_class_balance", "fixed_test_class_balance"):
        if balance_key in payload:
            flatten_dict(balance_key, payload[balance_key], row)

    return [row]


def summarize_csv_artifact(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        row_count = sum(1 for _ in reader)
    return [
        {
            "summary_source": str(path),
            "summary_format": "csv",
            "csv_row_count": row_count,
            "csv_columns": json.dumps(fieldnames, ensure_ascii=False),
        }
    ]


def summarize_artifact(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return summarize_json_metrics(path)
    if suffix in {".pkl", ".p"}:
        return summarize_pickle_metrics(path)
    if suffix == ".csv":
        return summarize_csv_artifact(path)
    return [{"summary_source": str(path), "summary_format": "unknown"}]


def parse_metrics_from_log(stdout_text: str) -> dict[str, Any]:
    patterns = {
        "log_accuracy": r"Accuracy:\s*([0-9]+(?:\.[0-9]+)?)",
        "log_auroc": r"AUROC?:\s*([0-9]+(?:\.[0-9]+)?)",
        "log_bacc": r"Bacc:\s*([0-9]+(?:\.[0-9]+)?)",
        "log_f1": r"F1(?:-score)?:\s*([0-9]+(?:\.[0-9]+)?)",
        "log_precision": r"Precision:\s*([0-9]+(?:\.[0-9]+)?)",
        "log_recall": r"Recall:\s*([0-9]+(?:\.[0-9]+)?)",
    }
    extracted: dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout_text)
        if match:
            extracted[key] = float(match.group(1))
    confusion_match = re.search(r"Confusion Matrix(?::| promedio:)\s*(\[[^\n]+\])", stdout_text)
    if confusion_match:
        extracted["log_confusion_matrix"] = confusion_match.group(1)
    return extracted


def build_base_row(
    method_name: str,
    dataset_name: str,
    run_id: str,
    status: str,
    exit_code: int | None,
    command: list[str],
    cwd: Path,
    run_log_path: Path,
) -> dict[str, Any]:
    return {
        "method_name": method_name,
        "dataset_name": dataset_name,
        "run_id": run_id,
        "status": status,
        "exit_code": exit_code,
        "cwd": str(cwd),
        "command": json.dumps(command, ensure_ascii=False),
        "run_log": str(run_log_path),
    }


def collect_summary_rows(
    run_dir: Path,
    method_name: str,
    dataset_name: str,
    run_id: str,
    status: str,
    exit_code: int | None,
    command: list[str],
    cwd: Path,
    run_log_path: Path,
    summary_patterns: list[str],
    run_log_text: str,
) -> list[dict[str, Any]]:
    base_row = build_base_row(
        method_name=method_name,
        dataset_name=dataset_name,
        run_id=run_id,
        status=status,
        exit_code=exit_code,
        command=command,
        cwd=cwd,
        run_log_path=run_log_path,
    )

    rows: list[dict[str, Any]] = []
    seen_files: set[Path] = set()
    for pattern in summary_patterns:
        for resolved_path in sorted(run_dir.glob(pattern)):
            if resolved_path in seen_files or not resolved_path.is_file():
                continue
            seen_files.add(resolved_path)
            for summary_row in summarize_artifact(resolved_path):
                merged = dict(base_row)
                merged.update(summary_row)
                rows.append(merged)

    if not rows:
        merged = dict(base_row)
        merged.update(parse_metrics_from_log(run_log_text))
        rows.append(merged)

    return rows


def rewrite_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_rows(csv_path: Path, new_rows: list[dict[str, Any]]) -> None:
    existing_rows: list[dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            existing_rows = list(csv.DictReader(handle))
    rewrite_csv(csv_path, [*existing_rows, *new_rows])


def update_global_index(results_dir: Path) -> None:
    collected_rows: list[dict[str, Any]] = []
    for summary_path in sorted(results_dir.glob("*/summary.csv")):
        with summary_path.open("r", newline="", encoding="utf-8") as handle:
            collected_rows.extend(csv.DictReader(handle))
    if collected_rows:
        rewrite_csv(results_dir / "all_runs.csv", collected_rows)


def run_method(
    method_config: dict,
    dataset_name: str,
    dataset_config: dict,
    results_dir: Path,
    rebuild: bool,
    dry_run: bool,
) -> list[dict]:
    execution = resolve_execution_context(
        method_config=method_config,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        results_dir=results_dir,
    )

    build_docker_image(
        method_config=method_config,
        template_context=execution["template_context"],
        rebuild=rebuild,
        dry_run=dry_run,
    )
    command = build_docker_run_command(
        method_config=method_config,
        template_context=execution["template_context"],
    )

    if dry_run:
        print(f"[dry-run] run: {join_command(command)}")
        summary_rows = collect_summary_rows(
            run_dir=execution["run_dir"],
            method_name=execution["method_name"],
            dataset_name=dataset_name,
            run_id=execution["run_id"],
            status="dry_run",
            exit_code=None,
            command=command,
            cwd=execution["cwd"],
            run_log_path=execution["run_log_path"],
            summary_patterns=[],
            run_log_text="",
        )
        cleanup_run_directory(execution)
        return summary_rows

    env = prepare_isolated_env(os.environ, execution["isolation_dir"])
    combined_chunks: list[str] = []
    with execution["run_log_path"].open("w", encoding="utf-8") as run_log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(execution["cwd"]),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            run_log_handle.write(line)
            run_log_handle.flush()
            combined_chunks.append(line)

        process.wait()

    combined_log = "".join(combined_chunks)
    if not combined_log and execution["run_log_path"].exists():
        combined_log = execution["run_log_path"].read_text(encoding="utf-8")

    status = "ok" if process.returncode == 0 else "failed"
    summary_rows = collect_summary_rows(
        run_dir=execution["run_dir"],
        method_name=execution["method_name"],
        dataset_name=dataset_name,
        run_id=execution["run_id"],
        status=status,
        exit_code=process.returncode,
        command=command,
        cwd=execution["cwd"],
        run_log_path=execution["run_log_path"],
        summary_patterns=execution["summary_patterns"],
        run_log_text=combined_log,
    )
    cleanup_run_directory(execution)
    return summary_rows


def cleanup_run_directory(execution: dict) -> None:
    for key in ("outputs_dir", "isolation_dir"):
        target = execution.get(key)
        if isinstance(target, Path) and target.exists():
            shutil.rmtree(target, ignore_errors=True)

    workspace_root = execution.get("workspace_root")
    if isinstance(workspace_root, Path) and workspace_root.exists() and workspace_root.name == "_workspace":
        shutil.rmtree(workspace_root, ignore_errors=True)


def main() -> int:
    args = parse_args()
    _ = resolve_python(args.python)
    catalog = load_catalog(args.catalog.resolve())

    if args.list:
        list_options(catalog)
        return 0

    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset
    dataset_config = catalog["datasets"].get(dataset_name)
    if dataset_config is None:
        raise SystemExit(f"Dataset desconocido: {dataset_name}")

    methods = select_methods(catalog, dataset_name, args.method)
    methods, skipped_methods = filter_methods_inside_final(methods)

    print(f"Dataset: {dataset_name}")
    print(f"Metodos a correr: {', '.join(method['name'] for method in methods)}")
    for method_name, missing_path in skipped_methods:
        print(f"[skip] {method_name}: no existe {missing_path}")

    exit_code = 0
    for method_config in methods:
        print()
        print(f"=== Ejecutando {method_config['name']} sobre {dataset_name} ===")
        summary_rows = run_method(
            method_config=method_config,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            results_dir=results_dir,
            rebuild=args.rebuild,
            dry_run=args.dry_run,
        )
        method_slug = method_config["name"]
        append_rows(results_dir / method_slug / "summary.csv", summary_rows)
        if any(row.get("status") == "failed" for row in summary_rows):
            exit_code = 1

    update_global_index(results_dir)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
