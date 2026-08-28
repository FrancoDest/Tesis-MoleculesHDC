"""Orquestador simple para:
1) leer embeddings mol2vec ya generados,
2) hacer split 80/20 del BBBP,
3) ajustar JL solo con train,
4) convertir la proyeccion JL a hipervectores HDC bipolares,
5) evaluar sobre el 20% restante.
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

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
    convert_hypervectors,
    score_for_predictions_csv,
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
DEFAULT_MOL2VEC_MODEL = "/opt/mol2vec/examples/models/model_300dim.pkl"
MOL2VEC_EXAMPLE_MODEL_URL = (
    "https://raw.githubusercontent.com/samoturk/mol2vec/master/"
    "examples/models/model_300dim.pkl"
)


def resolve_n_jobs(n_jobs: int, num_rows: int = 0, jl_dim: int = 0) -> int:
    """Si n_jobs no fue fijado explicitamente, usa todos los nucleos
    disponibles pero acotado por la memoria: cada worker del ProcessPoolExecutor
    materializa una matriz densa (train+test) de num_rows x jl_dim floats de 8
    bytes, y con datasets grandes (ej. hiv, ~41k moleculas) 32 workers en
    paralelo se comen la RAM del container y matan el pool (BrokenProcessPool)."""
    if n_jobs > 0:
        return n_jobs

    cpu_jobs = os.cpu_count() or 1
    if psutil is None or num_rows <= 0 or jl_dim <= 0:
        return cpu_jobs

    bytes_per_worker = num_rows * jl_dim * 8
    safety_fraction = 0.6
    available_bytes = psutil.virtual_memory().available
    memory_jobs = max(1, int((available_bytes * safety_fraction) // bytes_per_worker))
    return max(1, min(cpu_jobs, memory_jobs))


_WORKER_EMBEDDING_ROWS: List[EmbeddingRow] | None = None
_WORKER_EMBEDDINGS: np.ndarray | None = None
_WORKER_LABELS: np.ndarray | None = None
_WORKER_GROUPS: np.ndarray | None = None


def _init_iteration_worker(
    embedding_rows: List[EmbeddingRow], embeddings: np.ndarray, labels: np.ndarray, groups: np.ndarray | None = None
) -> None:
    global _WORKER_EMBEDDING_ROWS, _WORKER_EMBEDDINGS, _WORKER_LABELS, _WORKER_GROUPS
    _WORKER_EMBEDDING_ROWS = embedding_rows
    _WORKER_EMBEDDINGS = embeddings
    _WORKER_LABELS = labels
    _WORKER_GROUPS = groups


def _run_single_iteration(task: tuple[int, int, float, int, bool]) -> dict[str, object]:
    """Corre un split 80/20 + JL + SGD completo. Independiente entre iteraciones,
    por lo que se puede ejecutar en un proceso worker separado."""
    iteration, iteration_seed, test_size, jl_dim, keep_projection = task
    assert _WORKER_EMBEDDING_ROWS is not None and _WORKER_EMBEDDINGS is not None and _WORKER_LABELS is not None

    process = psutil.Process() if psutil is not None else None
    start_time = time.time()
    cpu_time_start = get_process_cpu_time_seconds(process) if process is not None else None

    if _WORKER_GROUPS is None:
        train_rows, test_rows, X_train, X_test, y_train, y_test = train_test_split(
            _WORKER_EMBEDDING_ROWS,
            _WORKER_EMBEDDINGS,
            _WORKER_LABELS,
            test_size=test_size,
            random_state=iteration_seed,
            stratify=_WORKER_LABELS,
        )
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=iteration_seed)
        train_idx, test_idx = next(splitter.split(_WORKER_EMBEDDINGS, _WORKER_LABELS, groups=_WORKER_GROUPS))
        train_rows = [_WORKER_EMBEDDING_ROWS[i] for i in train_idx]
        test_rows = [_WORKER_EMBEDDING_ROWS[i] for i in test_idx]
        X_train, X_test = _WORKER_EMBEDDINGS[train_idx], _WORKER_EMBEDDINGS[test_idx]
        y_train, y_test = _WORKER_LABELS[train_idx], _WORKER_LABELS[test_idx]

    projection_model = fit_projection_model(
        embeddings=np.asarray(X_train, dtype=np.float32),
        jl_dim=jl_dim,
        random_state=iteration_seed,
    )
    train_jl = apply_projection_model(projection_model, np.asarray(X_train, dtype=np.float32))
    test_jl = apply_projection_model(projection_model, np.asarray(X_test, dtype=np.float32))

    train_hdc = convert_hypervectors(np.asarray(train_jl))
    test_hdc = convert_hypervectors(np.asarray(test_jl))

    metrics, predicted_labels, predicted_proba, timing = train_and_evaluate_projection_classifier(
        train_features=np.asarray(train_hdc, dtype=np.float32),
        test_features=np.asarray(test_hdc, dtype=np.float32),
        train_labels=np.asarray(y_train, dtype=np.int64),
        test_labels=np.asarray(y_test, dtype=np.int64),
        random_state=iteration_seed,
    )
    predicted_scores = score_for_predictions_csv(predicted_proba, predicted_labels)

    elapsed = time.time() - start_time
    cpu_time_seconds, memory_percent, memory_mb = collect_process_metrics(process, cpu_time_start)

    prediction_rows = [
        {
            "iteration": iteration + 1,
            "random_state": iteration_seed,
            "id": row["id"],
            "smiles": row["smiles"],
            "y_true": int(true_value),
            "y_pred": int(pred_value),
            "y_score": float(score_value),
        }
        for row, true_value, pred_value, score_value in zip(
            test_rows, y_test, predicted_labels, predicted_scores
        )
    ]

    result: dict[str, object] = {
        "iteration": iteration,
        "iteration_seed": iteration_seed,
        "metrics": metrics,
        "timing": timing,
        "elapsed_seconds": elapsed,
        "cpu_time_seconds": cpu_time_seconds,
        "memory_percent": memory_percent,
        "memory_mb": memory_mb,
        "train_size": len(y_train),
        "test_size": len(y_test),
        "prediction_rows": prediction_rows,
    }
    if keep_projection:
        result["train_rows"] = train_rows
        result["test_rows"] = test_rows
        result["train_jl"] = train_jl
        result["test_jl"] = test_jl
        result["train_hdc"] = train_hdc
        result["test_hdc"] = test_hdc
    return result


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
    metrics_tracker["accuracy_list"].append(metrics["accuracy"])
    metrics_tracker["auroc_list"].append(metrics["roc_auc"])
    metrics_tracker["bacc_list"].append(metrics["balanced_accuracy"])
    metrics_tracker["f1_list"].append(metrics["f1"])
    metrics_tracker["precision_list"].append(metrics["precision"])
    metrics_tracker["recall_list"].append(metrics["recall"])
    metrics_tracker["confusion_matrices"].append(metrics["confusion_matrix"])
    metrics_tracker["random_states"].append(iteration_seed)
    metrics_tracker["cpu_time_seconds_list"].append(cpu_time_seconds)
    metrics_tracker["memory_percent_list"].append(memory_percent)
    metrics_tracker["memory_mb_list"].append(memory_mb)
    metrics_tracker["elapsed_seconds_list"].append(elapsed_seconds)
    metrics_tracker["training_seconds_list"].append(timing["training_seconds"])
    metrics_tracker["testing_seconds_list"].append(timing["testing_seconds"])


def std_metric(values: List[float]) -> float:
    """Desviacion estandar muestral (ddof=1). Con 1 sola iteracion no hay
    variabilidad que medir, asi que devuelve 0.0 en vez de dividir por 0."""
    clean_values = [value for value in values if value is not None]
    if len(clean_values) < 2:
        return 0.0
    return float(np.std(np.asarray(clean_values, dtype=np.float64), ddof=1))


def print_iteration_result(iteration: int, total: int, result: dict[str, object]) -> None:
    metrics = result["metrics"]
    cpu_time_seconds = result["cpu_time_seconds"] or 0.0
    memory_mb = result["memory_mb"] or 0.0
    print(
        f"[{iteration}/{total}] accuracy={metrics['accuracy']:.4f} "
        f"auroc={metrics['roc_auc']:.4f} balanced_accuracy={metrics['balanced_accuracy']:.4f} "
        f"f1={metrics['f1']:.4f} precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
        f"random_state={result['iteration_seed']} cpu_time={cpu_time_seconds:.2f}s "
        f"memory={memory_mb:.1f}MB elapsed={result['elapsed_seconds']:.2f}s"
    )


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
    best_run = metrics_payload["best_run"]
    worst_run = metrics_payload["worst_run"]
    average_metrics = metrics_payload["average_metrics"]
    average_resources = metrics_payload["average_resources"]
    timing_summary = metrics_payload["timing_summary"]

    print()
    print("Best AUROC run:")
    print("Accuracy:", best_run["accuracy"])
    print("AUROC:", best_run["roc_auc"])
    print("Bacc:", best_run["balanced_accuracy"])
    print("F1:", best_run["f1"])
    print("Precision:", best_run["precision"])
    print("Recall:", best_run["recall"])
    print("Confusion Matrix:", best_run["confusion_matrix"])
    print("Random State:", best_run["random_state"])

    print()
    print("Worst AUROC run:")
    print("Accuracy:", worst_run["accuracy"])
    print("AUROC:", worst_run["roc_auc"])
    print("Bacc:", worst_run["balanced_accuracy"])
    print("F1:", worst_run["f1"])
    print("Precision:", worst_run["precision"])
    print("Recall:", worst_run["recall"])
    print("Confusion Matrix:", worst_run["confusion_matrix"])
    print("Random State:", worst_run["random_state"])

    print()
    print(f"Average over {iteration_count} runs")
    print("Accuracy:", average_metrics["accuracy"])
    print("AUROC:", average_metrics["roc_auc"])
    print("Bacc:", average_metrics["balanced_accuracy"])
    print("F1:", average_metrics["f1"])
    print("Precision:", average_metrics["precision"])
    print("Recall:", average_metrics["recall"])
    print()
    print(f"Standard deviation over {iteration_count} runs")
    print("Accuracy:", std_metric(metrics_tracker["accuracy_list"]))
    print("AUROC:", std_metric(metrics_tracker["auroc_list"]))
    print("Bacc:", std_metric(metrics_tracker["bacc_list"]))
    print("F1:", std_metric(metrics_tracker["f1_list"]))
    print("Precision:", std_metric(metrics_tracker["precision_list"]))
    print("Recall:", std_metric(metrics_tracker["recall_list"]))
    print("Average Confusion Matrix:", average_metrics["confusion_matrix"])
    if average_resources["cpu_time_seconds"] is not None:
        print("Average CPU Total Time: {:.2f}s".format(average_resources["cpu_time_seconds"]))
    else:
        print("Average CPU Total Time: N/A")

    if average_resources["memory_percent"] is not None and average_resources["memory_mb"] is not None:
        print(
            "Average Memory Usage: {:.2f}% ({:.2f} MB)".format(
                average_resources["memory_percent"],
                average_resources["memory_mb"],
            )
        )
        print("Peak Memory Usage: {:.2f} MB".format(average_resources["peak_memory_mb"]))
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
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help=(
            "Nucleos a usar para correr las iteraciones 80/20 en paralelo. "
            "0 (default) usa todos los nucleos disponibles."
        ),
    )
    parser.add_argument(
        "--frac-a-column",
        default=None,
        help=(
            "Columna opcional en --labels-csv con una fraccion de composicion "
            ". Si se pasa junto con --frac-b-column, se suma "
            "como feature numerica extra al embedding antes de proyectar con JL."
        ),
    )
    parser.add_argument("--frac-b-column", default=None, help="Ver --frac-a-column.")
    parser.add_argument(
        "--group-column",
        default=None,
        help=(
            "Columna opcional en --labels-csv para agrupar el split train/test "
            ": garantiza que un mismo grupo no quede "
            "repartido entre train y test."
        ),
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


def load_extra_values_by_identifier(path: str, id_column: str, columns: list[str]) -> Dict[str, Dict[str, str]]:
    """Lee columnas auxiliares arbitrarias (ej. frac_a/frac_b/poly_id de un dataset con composicion)
    indexadas por id, sin castear -- el llamador decide el tipo."""
    values_by_identifier: Dict[str, Dict[str, str]] = {}
    if not columns:
        return values_by_identifier

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"El archivo {path} no tiene encabezados.")
        missing = [column for column in columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"No encontre estas columnas en {path}: {missing}")

        for row in reader:
            row_id = (row.get(id_column) or "").strip()
            if not row_id:
                continue
            values_by_identifier[row_id] = {column: row.get(column, "") for column in columns}

    return values_by_identifier


def merge_embeddings_with_labels(
    embedding_rows: List[EmbeddingRow],
    embedding_matrix: np.ndarray,
    labels_by_identifier: Dict[str, int],
    extra_values_by_identifier: Dict[str, Dict[str, str]] | None = None,
    frac_columns: Tuple[str, str] | None = None,
    group_column: str | None = None,
) -> Tuple[List[EmbeddingRow], np.ndarray, np.ndarray, np.ndarray | None]:
    """Conserva solo filas de embeddings que tienen label en BBBP. Si se pasan
    frac_columns, las suma como features extra al embedding; si se
    pasa group_column, devuelve el array de grupos para split agrupado."""
    matched_rows: List[EmbeddingRow] = []
    matched_vectors: List[np.ndarray] = []
    matched_labels: List[int] = []
    matched_fracs: List[List[float]] = []
    matched_groups: List[str] = []
    extra_values_by_identifier = extra_values_by_identifier or {}

    for row, vector in zip(embedding_rows, embedding_matrix):
        row_id = row["id"]
        if row_id not in labels_by_identifier:
            continue
        extra = extra_values_by_identifier.get(row_id, {})
        if frac_columns and (frac_columns[0] not in extra or frac_columns[1] not in extra):
            continue
        matched_rows.append(row)
        matched_vectors.append(vector)
        matched_labels.append(labels_by_identifier[row_id])
        if frac_columns:
            matched_fracs.append([float(extra[frac_columns[0]]), float(extra[frac_columns[1]])])
        if group_column:
            matched_groups.append(extra.get(group_column, row_id))

    if not matched_vectors:
        raise ValueError("No quedaron filas validas al unir embeddings con labels.")

    vectors = np.asarray(matched_vectors, dtype=np.float32)
    if frac_columns:
        vectors = np.concatenate([vectors, np.asarray(matched_fracs, dtype=np.float32)], axis=1)
    groups = np.asarray(matched_groups) if group_column else None

    return matched_rows, vectors, np.asarray(matched_labels, dtype=np.int64), groups


def build_output_file_map(embeddings_csv: str, output_dir: str) -> Dict[str, str]:
    """Deriva nombres de salida estables a partir del CSV de embeddings."""
    stem = Path(embeddings_csv).stem
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "train_jl": str(root / f"{stem}_train_jl.csv"),
        "test_jl": str(root / f"{stem}_test_jl.csv"),
        "train_hdc": str(root / f"{stem}_train_hdc.csv"),
        "test_hdc": str(root / f"{stem}_test_hdc.csv"),
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
    if not Path(model_path).is_file():
        raise FileNotFoundError(
            "No encontre el modelo Mol2vec pretrained en "
            f"{model_path}. El runtime Docker descarga el ejemplo oficial desde "
            f"{MOL2VEC_EXAMPLE_MODEL_URL}; si corres fuera de Docker, pasa "
            "--mol2vec-model con una copia local de model_300dim.pkl."
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

    embedding_rows, embeddings = load_embedding_table(args.embeddings_csv)
    total_rows_original = len(embedding_rows)
    labels_by_identifier = load_labels_by_identifier(
        path=args.labels_csv,
        id_column=args.label_id_column,
        target_column=args.target_column,
    )
    frac_columns = (
        (args.frac_a_column, args.frac_b_column)
        if args.frac_a_column and args.frac_b_column
        else None
    )
    extra_columns = [column for column in (args.frac_a_column, args.frac_b_column, args.group_column) if column]
    extra_values_by_identifier = load_extra_values_by_identifier(
        path=args.labels_csv,
        id_column=args.label_id_column,
        columns=extra_columns,
    )
    embedding_rows, embeddings, labels, groups = merge_embeddings_with_labels(
        embedding_rows,
        embeddings,
        labels_by_identifier,
        extra_values_by_identifier=extra_values_by_identifier,
        frac_columns=frac_columns,
        group_column=args.group_column,
    )

    metrics_dict = create_metrics_tracker()
    prediction_rows: List[Dict[str, object]] = []
    first_train_size = None
    first_test_size = None

    print_run_configuration(args, len(labels))

    n_jobs = resolve_n_jobs(args.n_jobs, num_rows=len(labels), jl_dim=args.jl_dim)
    print(f"Usando {n_jobs} nucleos para correr {args.iterations} iteraciones en paralelo.")

    tasks = [
        (iteration, args.random_state + iteration, args.test_size, args.jl_dim, iteration == 0)
        for iteration in range(args.iterations)
    ]

    if n_jobs > 1 and len(tasks) > 1:
        results = []
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_iteration_worker,
            initargs=(embedding_rows, embeddings, labels, groups),
        ) as executor:
            futures = [executor.submit(_run_single_iteration, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                print_iteration_result(done, len(tasks), result)
    else:
        _init_iteration_worker(embedding_rows, embeddings, labels, groups)
        results = []
        for done, task in enumerate(tasks, start=1):
            result = _run_single_iteration(task)
            results.append(result)
            print_iteration_result(done, len(tasks), result)

    for result in sorted(results, key=lambda item: item["iteration"]):
        iteration = result["iteration"]
        record_iteration_metrics(
            metrics_tracker=metrics_dict,
            metrics=result["metrics"],
            timing=result["timing"],
            iteration_seed=result["iteration_seed"],
            elapsed_seconds=result["elapsed_seconds"],
            cpu_time_seconds=result["cpu_time_seconds"],
            memory_percent=result["memory_percent"],
            memory_mb=result["memory_mb"],
        )
        prediction_rows.extend(result["prediction_rows"])

        if iteration == 0:
            first_train_size = result["train_size"]
            first_test_size = result["test_size"]
            print_iteration_sizes(first_train_size, first_test_size)
            write_projected_features_csv(output_files["train_jl"], result["train_rows"], result["train_jl"], prefix="jl")
            write_projected_features_csv(output_files["test_jl"], result["test_rows"], result["test_jl"], prefix="jl")
            write_projected_features_csv(output_files["train_hdc"], result["train_rows"], result["train_hdc"], prefix="hv")
            write_projected_features_csv(output_files["test_hdc"], result["test_rows"], result["test_hdc"], prefix="hv")

    write_prediction_rows_csv(output_files["predictions"], prediction_rows)

    best_idx = int(np.argmax(metrics_dict["auroc_list"]))
    worst_idx = int(np.argmin(metrics_dict["auroc_list"]))
    avg_conf_matrix = compute_average_confusion_matrix(metrics_dict["confusion_matrices"])

    valid_cpu = [value for value in metrics_dict["cpu_time_seconds_list"] if value is not None]
    valid_mem_percent = [value for value in metrics_dict["memory_percent_list"] if value is not None]
    valid_mem_mb = [value for value in metrics_dict["memory_mb_list"] if value is not None]

    metrics_payload = {
        "pipeline": "mol2vec_jl",
        "dataset_path": args.labels_csv,
        "source_embeddings_csv": args.embeddings_csv,
        "source_labels_csv": args.labels_csv,
        "total_rows_original": int(total_rows_original),
        "total_rows_valid_rdkit": int(len(labels)),
        "train_size": int(first_train_size) if first_train_size is not None else None,
        "test_size": int(first_test_size) if first_test_size is not None else None,
        "iterations": args.iterations,
        "test_fraction": args.test_size,
        "random_state_base": args.random_state,
        "jl_dim": args.jl_dim,
        "train_size_first_iteration": int(first_train_size) if first_train_size is not None else None,
        "test_size_first_iteration": int(first_test_size) if first_test_size is not None else None,
        "best_run": {
            "iteration": best_idx + 1,
            "random_state": metrics_dict["random_states"][best_idx],
            "accuracy": metrics_dict["accuracy_list"][best_idx],
            "roc_auc": metrics_dict["auroc_list"][best_idx],
            "balanced_accuracy": metrics_dict["bacc_list"][best_idx],
            "f1": metrics_dict["f1_list"][best_idx],
            "precision": metrics_dict["precision_list"][best_idx],
            "recall": metrics_dict["recall_list"][best_idx],
            "confusion_matrix": metrics_dict["confusion_matrices"][best_idx],
            "cpu_time_seconds": metrics_dict["cpu_time_seconds_list"][best_idx],
            "memory_percent": metrics_dict["memory_percent_list"][best_idx],
            "memory_mb": metrics_dict["memory_mb_list"][best_idx],
            "elapsed_seconds": metrics_dict["elapsed_seconds_list"][best_idx],
        },
        "worst_run": {
            "iteration": worst_idx + 1,
            "random_state": metrics_dict["random_states"][worst_idx],
            "accuracy": metrics_dict["accuracy_list"][worst_idx],
            "roc_auc": metrics_dict["auroc_list"][worst_idx],
            "balanced_accuracy": metrics_dict["bacc_list"][worst_idx],
            "f1": metrics_dict["f1_list"][worst_idx],
            "precision": metrics_dict["precision_list"][worst_idx],
            "recall": metrics_dict["recall_list"][worst_idx],
            "confusion_matrix": metrics_dict["confusion_matrices"][worst_idx],
            "cpu_time_seconds": metrics_dict["cpu_time_seconds_list"][worst_idx],
            "memory_percent": metrics_dict["memory_percent_list"][worst_idx],
            "memory_mb": metrics_dict["memory_mb_list"][worst_idx],
            "elapsed_seconds": metrics_dict["elapsed_seconds_list"][worst_idx],
        },
        "average_metrics": {
            "accuracy": float(np.mean(metrics_dict["accuracy_list"])),
            "roc_auc": float(np.mean(metrics_dict["auroc_list"])),
            "balanced_accuracy": float(np.mean(metrics_dict["bacc_list"])),
            "f1": float(np.mean(metrics_dict["f1_list"])),
            "precision": float(np.mean(metrics_dict["precision_list"])),
            "recall": float(np.mean(metrics_dict["recall_list"])),
            "confusion_matrix": avg_conf_matrix,
        },
        "average_resources": {
            "cpu_time_seconds": float(np.mean(valid_cpu)) if valid_cpu else None,
            "memory_percent": float(np.mean(valid_mem_percent)) if valid_mem_percent else None,
            "memory_mb": float(np.mean(valid_mem_mb)) if valid_mem_mb else None,
            "peak_memory_mb": float(np.max(valid_mem_mb)) if valid_mem_mb else None,
            "elapsed_seconds": float(np.mean(metrics_dict["elapsed_seconds_list"])),
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
