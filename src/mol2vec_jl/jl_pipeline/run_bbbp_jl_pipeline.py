"""Orquestador simple para:
1) leer embeddings mol2vec ya generados,
2) hacer split 80/20 del BBBP,
3) ajustar JL solo con train,
4) evaluar sobre el 20% restante.
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

ORIGINAL_DIR = Path(__file__).resolve().parents[1] / "original"
if str(ORIGINAL_DIR) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_DIR))

from gensim.models import word2vec
import features

try:
    import psutil
except ImportError:  # pragma: no cover - fallback para entornos minimos
    psutil = None

from core import (
    compute_average_confusion_matrix,
    train_and_evaluate_projection_classifier,
    write_metrics_summary_json,
    write_prediction_rows_csv,
)
from johnson_lindenstrauss import (
    apply_projection_model,
    fit_projection_model,
    load_embedding_table,
    write_projected_features_csv,
)

EmbeddingRow = Dict[str, str]


DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 800
DEFAULT_JL_DIM = 10048
DEFAULT_ITERATIONS = 100
DEFAULT_MOL2VEC_MODEL = "/tesis/mol2vec/examples/models/model_300dim.pkl"


def ensure_legacy_word2vec_subscriptable() -> None:
    """Compatibilidad con mol2vec original en gensim nuevos."""
    if getattr(word2vec.Word2Vec, "__getitem__", None) is None:
        def _getitem(self, key):
            return self.wv[key]
        word2vec.Word2Vec.__getitem__ = _getitem


def get_process_cpu_time_seconds(process) -> float:
    cpu_times = process.cpu_times()
    return float(cpu_times.user + cpu_times.system)


def create_metrics_tracker() -> Dict[str, List[object]]:
    return {
        "accuracy_list": [],
        "auroc_list": [],
        "bacc_list": [],
        "f1_list": [],
        "precision_list": [],
        "recall_list": [],
        "confusion_matrices": [],
        "random_states": [],
        "cpu_time_seconds_list": [],
        "memory_percent_list": [],
        "memory_mb_list": [],
        "elapsed_seconds_list": [],
        "training_seconds_list": [],
        "testing_seconds_list": [],
    }


def collect_process_metrics(process, cpu_time_start: float | None) -> tuple[float | None, float | None, float | None]:
    if process is None or cpu_time_start is None:
        return None, None, None

    cpu_time_seconds = max(0.0, get_process_cpu_time_seconds(process) - cpu_time_start)
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    memory_mb = memory_info.rss / 1024 / 1024
    return cpu_time_seconds, memory_percent, memory_mb


def record_iteration_metrics(
    metrics_tracker: Dict[str, List[object]],
    metrics: Dict[str, object],
    timing: Dict[str, float],
    iteration_seed: int,
    elapsed_seconds: float,
    cpu_time_seconds: float | None,
    memory_percent: float | None,
    memory_mb: float | None,
) -> None:
    metrics_tracker["accuracy_list"].append(metrics["Accuracy"])
    metrics_tracker["auroc_list"].append(metrics["AUROC"])
    metrics_tracker["bacc_list"].append(metrics["Bacc"])
    metrics_tracker["f1_list"].append(metrics["F1"])
    metrics_tracker["precision_list"].append(metrics["Precision"])
    metrics_tracker["recall_list"].append(metrics["Recall"])
    metrics_tracker["confusion_matrices"].append(metrics["ConfusionMatrix"])
    metrics_tracker["random_states"].append(iteration_seed)
    metrics_tracker["cpu_time_seconds_list"].append(cpu_time_seconds)
    metrics_tracker["memory_percent_list"].append(memory_percent)
    metrics_tracker["memory_mb_list"].append(memory_mb)
    metrics_tracker["elapsed_seconds_list"].append(elapsed_seconds)
    metrics_tracker["training_seconds_list"].append(timing["training_seconds"])
    metrics_tracker["testing_seconds_list"].append(timing["testing_seconds"])


def print_run_configuration(args: argparse.Namespace, label_count: int) -> None:
    print(f"Valid rows: {label_count}")
    print(f"Iterations: {args.iterations}")
    print(f"Base random state: {args.random_state}")
    print(
        "Split per iteration: "
        f"{int((1 - args.test_size) * 100)}/{int(args.test_size * 100)} "
        "with base seed + iteration"
    )
    print(f"Fixed JL dimension: {args.jl_dim}")


def print_iteration_sizes(train_size: int, test_size: int) -> None:
    print(f"Train size (~80%): {train_size}")
    print(f"Test size (~20%): {test_size}")


def print_final_summary(metrics_payload: Dict[str, object], iteration_count: int, metrics_tracker: Dict[str, List[object]]) -> None:
    best_run = metrics_payload["best_iteration_by_auroc"]
    worst_run = metrics_payload["worst_iteration_by_auroc"]
    average_metrics = metrics_payload["average_metrics"]
    timing_summary = metrics_payload["timing_summary"]

    print()
    print("Best AUROC run:")
    print("Accuracy:", best_run["Accuracy"])
    print("AUROC:", best_run["AUROC"])
    print("Bacc:", best_run["Bacc"])
    print("F1:", best_run["F1"])
    print("Precision:", best_run["Precision"])
    print("Recall:", best_run["Recall"])
    print("Confusion Matrix:", best_run["ConfusionMatrix"])
    print("Random State:", best_run["random_state"])

    print()
    print("Worst AUROC run:")
    print("Accuracy:", worst_run["Accuracy"])
    print("AUROC:", worst_run["AUROC"])
    print("Bacc:", worst_run["Bacc"])
    print("F1:", worst_run["F1"])
    print("Precision:", worst_run["Precision"])
    print("Recall:", worst_run["Recall"])
    print("Confusion Matrix:", worst_run["ConfusionMatrix"])
    print("Random State:", worst_run["random_state"])

    print()
    print(f"Average over {iteration_count} runs")
    print("Accuracy:", average_metrics["Accuracy"])
    print("AUROC:", average_metrics["AUROC"])
    print("Bacc:", average_metrics["Bacc"])
    print("F1:", average_metrics["F1"])
    print("Precision:", average_metrics["Precision"])
    print("Recall:", average_metrics["Recall"])
    print("Average Confusion Matrix:", average_metrics["ConfusionMatrix"])
    if average_metrics["AverageCPUTotalTimeSeconds"] is not None:
        print("Average CPU Total Time: {:.2f}s".format(average_metrics["AverageCPUTotalTimeSeconds"]))
    else:
        print("Average CPU Total Time: N/A")

    if average_metrics["AverageMemoryPercent"] is not None and average_metrics["AverageMemoryMB"] is not None:
        print(
            "Average Memory Usage: {:.2f}% ({:.2f} MB)".format(
                average_metrics["AverageMemoryPercent"],
                average_metrics["AverageMemoryMB"],
            )
        )
        valid_memory_mb = [value for value in metrics_tracker["memory_mb_list"] if value is not None]
        print("Peak Memory Usage: {:.2f} MB".format(float(np.max(valid_memory_mb))))
    else:
        print("Average Memory Usage: N/A")
        print("Peak Memory Usage: N/A")

    print()
    print("Total timings:")
    print("Total time: {:.2f}s".format(timing_summary["total_wall_clock_seconds"]))
    print("Total training time: {:.2f}s".format(timing_summary["total_training_seconds"]))
    print("Total testing time: {:.2f}s".format(timing_summary["total_testing_seconds"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Usa embeddings mol2vec ya generados, hace split 80/20, ajusta "
            "Johnson-Lindenstrauss con train y evalua en el 20% restante."
        )
    )
    parser.add_argument(
        "--embeddings-csv",
        default="/run_outputs/artifacts/bbbp_mol2vec_features.csv",
        help="CSV de embeddings generado por mol2vec featurize.",
    )
    parser.add_argument(
        "--labels-csv",
        default="/tesis/Tesis-PolymerHDC/data/bbbp/raw/BBBP.csv",
        help="CSV original BBBP con la columna target.",
    )
    parser.add_argument(
        "--smiles-column",
        default="smiles",
        help="Nombre de la columna SMILES en el CSV de labels.",
    )
    parser.add_argument(
        "--output-dir",
        default="/run_outputs/artifacts",
        help="Carpeta donde se guardan train_jl, test_jl, metricas y predicciones.",
    )
    parser.add_argument(
        "--target-column",
        default="p_np",
        help="Nombre de la columna target en BBBP.csv.",
    )
    parser.add_argument(
        "--label-id-column",
        default="num",
        help="Nombre de la columna id en BBBP.csv.",
    )
    parser.add_argument(
        "--mol2vec-model",
        default=DEFAULT_MOL2VEC_MODEL,
        help="Modelo mol2vec .pkl usado si hay que generar embeddings.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Seed fija para el split 80/20 y la proyeccion JL.",
    )
    parser.add_argument(
        "--jl-dim",
        type=int,
        default=DEFAULT_JL_DIM,
        help="Dimension de la proyeccion Johnson-Lindenstrauss.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Cantidad de corridas repetidas 80/20.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraccion usada para test. 0.2 equivale a 80/20.",
    )
    return parser.parse_args()


def parse_binary_target(value: str) -> int:
    """Convierte el target a entero binario."""
    return int(float(str(value).strip()))


def load_labels_by_identifier(
    path: str,
    id_column: str,
    target_column: str,
) -> Dict[str, int]:
    """Carga labels desde BBBP.csv indexando por id original."""
    labels_by_identifier: Dict[str, int] = {}

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"El archivo {path} no tiene encabezados.")
        if id_column not in reader.fieldnames:
            raise ValueError(f"No encontre la columna '{id_column}' en {path}.")
        if target_column not in reader.fieldnames:
            raise ValueError(f"No encontre la columna '{target_column}' en {path}.")

        for row in reader:
            row_id = (row.get(id_column) or "").strip()
            target_value = row.get(target_column)
            if not row_id or target_value in (None, ""):
                continue
            labels_by_identifier[row_id] = parse_binary_target(target_value)

    if not labels_by_identifier:
        raise ValueError("No pude leer labels validos desde el CSV de BBBP.")

    return labels_by_identifier


def merge_embeddings_with_labels(
    embedding_rows: List[EmbeddingRow],
    embedding_matrix: np.ndarray,
    labels_by_identifier: Dict[str, int],
) -> Tuple[List[EmbeddingRow], np.ndarray, np.ndarray]:
    """Conserva solo filas de embeddings que tienen label en BBBP."""
    matched_rows: List[EmbeddingRow] = []
    matched_vectors: List[np.ndarray] = []
    matched_labels: List[int] = []

    for row, vector in zip(embedding_rows, embedding_matrix):
        row_id = row["id"]
        if row_id not in labels_by_identifier:
            continue
        matched_rows.append(row)
        matched_vectors.append(vector)
        matched_labels.append(labels_by_identifier[row_id])

    if not matched_vectors:
        raise ValueError("No quedaron filas validas al unir embeddings con labels.")

    return matched_rows, np.asarray(matched_vectors, dtype=np.float32), np.asarray(matched_labels, dtype=np.int64)


def build_output_file_map(embeddings_csv: str, output_dir: str) -> Dict[str, str]:
    """Deriva nombres de salida estables a partir del CSV de embeddings."""
    stem = Path(embeddings_csv).stem
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "train_jl": str(root / f"{stem}_train_jl.csv"),
        "test_jl": str(root / f"{stem}_test_jl.csv"),
        "metrics": str(root / f"{stem}_jl_metrics.json"),
        "predictions": str(root / f"{stem}_jl_test_predictions.csv"),
    }


def ensure_embeddings_exist(
    *,
    embeddings_csv: str,
    labels_csv: str,
    smiles_column: str,
    label_id_column: str,
    model_path: str,
) -> None:
    embeddings_path = Path(embeddings_csv)
    if embeddings_path.exists():
        return
    if not model_path:
        raise ValueError(
            "Embeddings file is missing and no mol2vec model was provided. "
            "Pass --mol2vec-model explicitly or generate embeddings beforehand."
        )

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    smiles_input_path = embeddings_path.with_suffix(".smi")

    with open(labels_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"El archivo {labels_csv} no tiene encabezados.")
        if smiles_column not in reader.fieldnames:
            raise ValueError(f"No encontre la columna '{smiles_column}' en {labels_csv}.")
        if label_id_column not in reader.fieldnames:
            raise ValueError(f"No encontre la columna '{label_id_column}' en {labels_csv}.")

        with smiles_input_path.open("w", encoding="utf-8") as out_handle:
            for row in reader:
                smiles = (row.get(smiles_column) or "").strip()
                row_id = (row.get(label_id_column) or "").strip()
                if not smiles or not row_id:
                    continue
                out_handle.write(f"{smiles}\t{row_id}\n")

    print(f"Generating mol2vec embeddings at: {embeddings_csv}")
    features.featurize(
        in_file=str(smiles_input_path),
        out_file=str(embeddings_path),
        model_path=model_path,
        r=1,
        uncommon="UNK",
    )


def main() -> None:
    overall_start = time.time()
    args = parse_args()
    ensure_legacy_word2vec_subscriptable()
    ensure_embeddings_exist(
        embeddings_csv=args.embeddings_csv,
        labels_csv=args.labels_csv,
        smiles_column=args.smiles_column,
        label_id_column=args.label_id_column,
        model_path=args.mol2vec_model,
    )
    output_files = build_output_file_map(args.embeddings_csv, args.output_dir)
    process = psutil.Process() if psutil is not None else None

    embedding_rows, embeddings = load_embedding_table(args.embeddings_csv)
    labels_by_identifier = load_labels_by_identifier(
        path=args.labels_csv,
        id_column=args.label_id_column,
        target_column=args.target_column,
    )
    embedding_rows, embeddings, labels = merge_embeddings_with_labels(
        embedding_rows,
        embeddings,
        labels_by_identifier,
    )

    metrics_dict = create_metrics_tracker()
    prediction_rows: List[Dict[str, object]] = []
    first_iteration_saved = False
    first_train_size = None
    first_test_size = None

    print_run_configuration(args, len(labels))

    for iteration in range(args.iterations):
        start_time = time.time()
        cpu_time_start = get_process_cpu_time_seconds(process) if process is not None else None
        iteration_seed = args.random_state + iteration

        train_rows, test_rows, X_train, X_test, y_train, y_test = train_test_split(
            embedding_rows,
            embeddings,
            labels,
            test_size=args.test_size,
            random_state=iteration_seed,
            stratify=labels,
        )

        projection_model = fit_projection_model(
            embeddings=np.asarray(X_train, dtype=np.float32),
            jl_dim=args.jl_dim,
            random_state=iteration_seed,
        )
        train_jl = apply_projection_model(projection_model, np.asarray(X_train, dtype=np.float32))
        test_jl = apply_projection_model(projection_model, np.asarray(X_test, dtype=np.float32))

        if not first_iteration_saved:
            write_projected_features_csv(output_files["train_jl"], train_rows, train_jl, prefix="jl")
            write_projected_features_csv(output_files["test_jl"], test_rows, test_jl, prefix="jl")
            first_iteration_saved = True

        metrics, predicted_labels, predicted_scores, timing = train_and_evaluate_projection_classifier(
            train_features=np.asarray(train_jl, dtype=np.float32),
            test_features=np.asarray(test_jl, dtype=np.float32),
            train_labels=np.asarray(y_train, dtype=np.int64),
            test_labels=np.asarray(y_test, dtype=np.int64),
            random_state=iteration_seed,
        )

        elapsed = time.time() - start_time
        cpu_time_seconds, memory_percent, memory_mb = collect_process_metrics(process, cpu_time_start)
        record_iteration_metrics(
            metrics_tracker=metrics_dict,
            metrics=metrics,
            timing=timing,
            iteration_seed=iteration_seed,
            elapsed_seconds=elapsed,
            cpu_time_seconds=cpu_time_seconds,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
        )

        if iteration == 0:
            first_train_size = len(y_train)
            first_test_size = len(y_test)
            print_iteration_sizes(len(y_train), len(y_test))

        for row, true_value, pred_value, score_value in zip(test_rows, y_test, predicted_labels, predicted_scores):
            prediction_rows.append(
                {
                    "iteration": iteration + 1,
                    "random_state": iteration_seed,
                    "id": row["id"],
                    "smiles": row["smiles"],
                    "y_true": int(true_value),
                    "y_pred": int(pred_value),
                    "y_score": float(score_value),
                }
            )

    write_prediction_rows_csv(output_files["predictions"], prediction_rows)

    best_idx = int(np.argmax(metrics_dict["auroc_list"]))
    worst_idx = int(np.argmin(metrics_dict["auroc_list"]))
    avg_conf_matrix = compute_average_confusion_matrix(metrics_dict["confusion_matrices"])

    valid_cpu = [value for value in metrics_dict["cpu_time_seconds_list"] if value is not None]
    valid_mem_percent = [value for value in metrics_dict["memory_percent_list"] if value is not None]
    valid_mem_mb = [value for value in metrics_dict["memory_mb_list"] if value is not None]

    metrics_payload = {
        "pipeline": f"mol2vec embeddings -> repeated 80/20 split -> JL(train only, dim {args.jl_dim}) -> SGDClassifier -> test",
        "source_embeddings_csv": args.embeddings_csv,
        "source_labels_csv": args.labels_csv,
        "total_rows_with_embeddings_and_labels": int(len(labels)),
        "iterations": args.iterations,
        "test_fraction": args.test_size,
        "random_state_base": args.random_state,
        "jl_dim": args.jl_dim,
        "train_size_first_iteration": int(first_train_size) if first_train_size is not None else None,
        "test_size_first_iteration": int(first_test_size) if first_test_size is not None else None,
        "best_iteration_by_auroc": {
            "iteration": best_idx + 1,
            "random_state": metrics_dict["random_states"][best_idx],
            "Accuracy": metrics_dict["accuracy_list"][best_idx],
            "AUROC": metrics_dict["auroc_list"][best_idx],
            "Bacc": metrics_dict["bacc_list"][best_idx],
            "F1": metrics_dict["f1_list"][best_idx],
            "Precision": metrics_dict["precision_list"][best_idx],
            "Recall": metrics_dict["recall_list"][best_idx],
            "ConfusionMatrix": metrics_dict["confusion_matrices"][best_idx],
        },
        "worst_iteration_by_auroc": {
            "iteration": worst_idx + 1,
            "random_state": metrics_dict["random_states"][worst_idx],
            "Accuracy": metrics_dict["accuracy_list"][worst_idx],
            "AUROC": metrics_dict["auroc_list"][worst_idx],
            "Bacc": metrics_dict["bacc_list"][worst_idx],
            "F1": metrics_dict["f1_list"][worst_idx],
            "Precision": metrics_dict["precision_list"][worst_idx],
            "Recall": metrics_dict["recall_list"][worst_idx],
            "ConfusionMatrix": metrics_dict["confusion_matrices"][worst_idx],
        },
        "average_metrics": {
            "Accuracy": float(np.mean(metrics_dict["accuracy_list"])),
            "AUROC": float(np.mean(metrics_dict["auroc_list"])),
            "Bacc": float(np.mean(metrics_dict["bacc_list"])),
            "F1": float(np.mean(metrics_dict["f1_list"])),
            "Precision": float(np.mean(metrics_dict["precision_list"])),
            "Recall": float(np.mean(metrics_dict["recall_list"])),
            "ConfusionMatrix": avg_conf_matrix,
            "AverageCPUTotalTimeSeconds": float(np.mean(valid_cpu)) if valid_cpu else None,
            "AverageMemoryPercent": float(np.mean(valid_mem_percent)) if valid_mem_percent else None,
            "AverageMemoryMB": float(np.mean(valid_mem_mb)) if valid_mem_mb else None,
            "AverageElapsedSeconds": float(np.mean(metrics_dict["elapsed_seconds_list"])),
        },
        "timing_summary": {
            "total_wall_clock_seconds": float(time.time() - overall_start),
            "total_training_seconds": float(sum(metrics_dict["training_seconds_list"])),
            "total_testing_seconds": float(sum(metrics_dict["testing_seconds_list"])),
        },
        "artifacts": output_files,
    }
    write_metrics_summary_json(output_files["metrics"], metrics_payload)

    print_final_summary(metrics_payload, args.iterations, metrics_dict)
    print(f"Metrics saved to: {output_files['metrics']}")
    print(f"Predictions saved to: {output_files['predictions']}")


if __name__ == "__main__":
    main()
