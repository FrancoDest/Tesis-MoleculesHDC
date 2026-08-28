"""Punto de entrada unico del proyecto.

Flujo cerrado:
1. leer CSV base
2. separar train/test 80/20
3. generar embeddings de train y test por separado
4. ajustar JL solo con train
5. transformar test con el mismo proyector JL
6. convertir ambos a HDC bipolar
7. entrenar con train HDC y evaluar con test HDC
"""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import psutil
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from core import (
    DEFAULT_ITERATIONS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    compute_average_confusion_matrix,
    generate_embeddings_from_rows,
    convert_hypervectors,
    get_process_cpu_time_seconds,
    load_dataset_rows,
    parse_binary_target,
    score_for_predictions_csv,
    train_and_evaluate_classifier,
    write_matrix_csv,
    write_prediction_rows_csv,
)
from johnson_lindenstrauss import apply_projection_model, fit_projection_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta el flujo completo CSV -> train/test split -> embeddings -> "
            "JL(train only) -> HDC -> evaluacion."
        )
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help=(
            "CSV base del dataset. Para BBBP conviene usar "
            "'/tesis/Tesis-PolymerHDC/data/bbbp/raw/BBBP_rdkit_valid.csv'."
        ),
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Columna target binaria.",
    )
    parser.add_argument(
        "--smiles-column",
        default="smiles",
        help="Nombre de la columna smiles en el dataset base.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint del encoder Mole-BERT. Si no lo pasas, el pipeline corta con error claro.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Carpeta donde se guardan los artefactos del pipeline.",
    )
    parser.add_argument(
        "--jl-dim",
        type=int,
        default=10048,
        help="Dimension de la proyeccion JL y, por extension, del HDC resultante.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Cantidad de corridas repetidas 80/20.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Semilla base para las corridas.",
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
            "Nucleos a usar para correr las iteraciones JL+HDC+SGD en paralelo. "
            "0 (default) usa todos los nucleos disponibles."
        ),
    )
    parser.add_argument(
        "--frac-a-column",
        default=None,
        help=(
            "Columna opcional en --input-csv con una fraccion de composicion "
            ". Si se pasa junto con --frac-b-column, se suma "
            "como feature numerica extra al embedding antes de proyectar con JL."
        ),
    )
    parser.add_argument("--frac-b-column", default=None, help="Ver --frac-a-column.")
    parser.add_argument(
        "--group-column",
        default=None,
        help=(
            "Columna opcional en --input-csv para agrupar el split train/test "
            ": garantiza que un mismo grupo no quede "
            "repartido entre train y test."
        ),
    )
    return parser.parse_args()


def resolve_n_jobs(n_jobs: int, n_rows: int = 0, jl_dim: int = 0) -> int:
    """Si no se pasa --n-jobs, limita los workers segun la RAM disponible en
    vez de usar todos los cores. Cada worker mantiene en memoria train_jl +
    test_jl (n_rows x jl_dim, float64) mas copias intermedias de la
    conversion a HDC bipolar y el entrenamiento SGD. Con jl_dim=10048 y
    datasets grandes (HIV ~41k filas) eso son >3GB por iteracion, y con
    varios workers corriendo la primera iteracion en simultaneo se llego a
    OOM incluso con 8 workers (BrokenProcessPool)."""
    cpu_workers = os.cpu_count() or 1
    if n_jobs > 0:
        return n_jobs
    if n_rows <= 0 or jl_dim <= 0:
        return min(8, cpu_workers)

    bytes_per_iteration = n_rows * jl_dim * 8
    per_worker_estimate = int(bytes_per_iteration * 1.8 + 256 * 1024 * 1024)
    try:
        usable_memory = psutil.virtual_memory().available * 0.5
    except Exception:
        return min(8, cpu_workers)

    mem_workers = max(1, int(usable_memory // per_worker_estimate))
    return max(1, min(cpu_workers, mem_workers, 8))


def build_output_file_map(input_csv: str, output_dir: str) -> dict[str, str]:
    """Deriva nombres de salida estables."""
    base_name = Path(input_csv).stem.lower()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "train_embeddings": str(root / f"{base_name}_train_embeddings.csv"),
        "train_jl": str(root / f"{base_name}_train_jl.csv"),
        "train_hdc": str(root / f"{base_name}_train_hdc.csv"),
        "test_embeddings": str(root / f"{base_name}_test_embeddings.csv"),
        "test_hdc": str(root / f"{base_name}_test_hdc.csv"),
        "metrics": str(root / f"{base_name}_hdc_metrics.json"),
        "predictions": str(root / f"{base_name}_hdc_test_predictions.csv"),
    }


def create_metrics_tracker() -> dict[str, list[object]]:
    return {
        "accuracy_list": [],
        "auroc_list": [],
        "bacc_list": [],
        "f1_list": [],
        "precision_list": [],
        "recall_list": [],
        "confusion_matrices": [],
        "random_states": [],
        "train_sizes": [],
        "test_sizes": [],
        "cpu_time_seconds_list": [],
        "memory_percent_list": [],
        "memory_mb_list": [],
        "elapsed_seconds_list": [],
        "training_seconds_list": [],
        "testing_seconds_list": [],
    }


def std_metric(values: list[float]) -> float:
    """Desviacion estandar muestral (ddof=1). Con 1 sola iteracion no hay
    variabilidad que medir, asi que devuelve 0.0 en vez de dividir por 0."""
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def print_iteration_result(iteration: int, total: int, result: dict[str, object]) -> None:
    metrics = result["metrics"]
    process_metrics = result["process_metrics"]
    print(
        f"[{iteration}/{total}] accuracy={metrics['accuracy']:.4f} "
        f"auroc={metrics['roc_auc']:.4f} balanced_accuracy={metrics['balanced_accuracy']:.4f} "
        f"f1={metrics['f1']:.4f} precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
        f"random_state={result['iteration_seed']} cpu_time={process_metrics['cpu_time_seconds']:.2f}s "
        f"memory={process_metrics['memory_mb']:.1f}MB elapsed={process_metrics['elapsed_seconds']:.2f}s"
    )


def print_run_configuration(args: argparse.Namespace, total_rows: int) -> None:
    print(f"Filas validas: {total_rows}")
    print(f"Iteraciones: {args.iterations}")
    print(f"Random state base: {args.random_state}")
    print(f"Split por iteracion: {int((1 - args.test_size) * 100)}/{int(args.test_size * 100)} con seed base + iteracion")
    print("JL: se ajusta solo con train y se aplica al test con el mismo proyector")


def print_iteration_sizes(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]]) -> None:
    print(f"Train size (80% aprox): {len(train_rows)}")
    print(f"Test size (20% aprox): {len(test_rows)}")


def record_iteration_metrics(
    tracker: dict[str, list[object]],
    metrics: dict[str, object],
    random_state: int,
    process_metrics: dict[str, float],
    timing: dict[str, float],
    train_size: int = 0,
    test_size: int = 0,
) -> None:
    tracker["accuracy_list"].append(metrics["accuracy"])
    tracker["auroc_list"].append(metrics["roc_auc"])
    tracker["bacc_list"].append(metrics["balanced_accuracy"])
    tracker["f1_list"].append(metrics["f1"])
    tracker["precision_list"].append(metrics["precision"])
    tracker["recall_list"].append(metrics["recall"])
    tracker["confusion_matrices"].append(metrics["confusion_matrix"])
    tracker["random_states"].append(random_state)
    tracker["train_sizes"].append(train_size)
    tracker["test_sizes"].append(test_size)
    tracker["cpu_time_seconds_list"].append(process_metrics["cpu_time_seconds"])
    tracker["memory_percent_list"].append(process_metrics["memory_percent"])
    tracker["memory_mb_list"].append(process_metrics["memory_mb"])
    tracker["elapsed_seconds_list"].append(process_metrics["elapsed_seconds"])
    tracker["training_seconds_list"].append(timing["training_seconds"])
    tracker["testing_seconds_list"].append(timing["testing_seconds"])


def print_final_summary(metrics_tracker: dict[str, list[object]], iterations: int) -> None:
    best_index = int(np.argmax(metrics_tracker["auroc_list"]))
    worst_index = int(np.argmin(metrics_tracker["auroc_list"]))
    average_confusion_matrix = compute_average_confusion_matrix(metrics_tracker["confusion_matrices"])

    print()
    print("Stats correspondientes al mejor AUROC:")
    print("Accuracy:", metrics_tracker["accuracy_list"][best_index])
    print("Auroc:", metrics_tracker["auroc_list"][best_index])
    print("Bacc:", metrics_tracker["bacc_list"][best_index])
    print("F1:", metrics_tracker["f1_list"][best_index])
    print("Precision:", metrics_tracker["precision_list"][best_index])
    print("Recall:", metrics_tracker["recall_list"][best_index])
    print("Confusion Matrix:", metrics_tracker["confusion_matrices"][best_index])
    print("Random State:", metrics_tracker["random_states"][best_index])

    print()
    print("Stats correspondientes al peor AUROC:")
    print("Accuracy:", metrics_tracker["accuracy_list"][worst_index])
    print("Auroc:", metrics_tracker["auroc_list"][worst_index])
    print("Bacc:", metrics_tracker["bacc_list"][worst_index])
    print("F1:", metrics_tracker["f1_list"][worst_index])
    print("Precision:", metrics_tracker["precision_list"][worst_index])
    print("Recall:", metrics_tracker["recall_list"][worst_index])
    print("Confusion Matrix:", metrics_tracker["confusion_matrices"][worst_index])
    print("Random State:", metrics_tracker["random_states"][worst_index])

    print()
    print(f"Promedio para {iterations} corridas")
    print("Accuracy:", float(np.mean(metrics_tracker["accuracy_list"])))
    print("Auroc:", float(np.mean(metrics_tracker["auroc_list"])))
    print("Bacc:", float(np.mean(metrics_tracker["bacc_list"])))
    print("F1:", float(np.mean(metrics_tracker["f1_list"])))
    print("Precision:", float(np.mean(metrics_tracker["precision_list"])))
    print("Recall:", float(np.mean(metrics_tracker["recall_list"])))
    print()
    print(f"Desviacion estandar para {iterations} corridas")
    print("Accuracy:", std_metric(metrics_tracker["accuracy_list"]))
    print("Auroc:", std_metric(metrics_tracker["auroc_list"]))
    print("Bacc:", std_metric(metrics_tracker["bacc_list"]))
    print("F1:", std_metric(metrics_tracker["f1_list"]))
    print("Precision:", std_metric(metrics_tracker["precision_list"]))
    print("Recall:", std_metric(metrics_tracker["recall_list"]))
    print("Confusion Matrix promedio:", average_confusion_matrix)
    print("Average CPU Total Time: {:.2f}s".format(float(np.mean(metrics_tracker["cpu_time_seconds_list"]))))
    print(
        "Average Memory Usage: {:.2f}% ({:.2f} MB)".format(
            float(np.mean(metrics_tracker["memory_percent_list"])),
            float(np.mean(metrics_tracker["memory_mb_list"])),
        )
    )
    print("Peak Memory Usage: {:.2f} MB".format(float(np.max(metrics_tracker["memory_mb_list"]))))


_WORKER_VALID_ROWS: list[dict[str, str]] | None = None
_WORKER_EMBEDDINGS: np.ndarray | None = None
_WORKER_LABELS: np.ndarray | None = None
_WORKER_GROUPS: np.ndarray | None = None


def _init_iteration_worker(
    valid_rows: list[dict[str, str]],
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | None = None,
) -> None:
    global _WORKER_VALID_ROWS, _WORKER_EMBEDDINGS, _WORKER_LABELS, _WORKER_GROUPS
    _WORKER_VALID_ROWS = valid_rows
    _WORKER_EMBEDDINGS = embeddings
    _WORKER_LABELS = labels
    _WORKER_GROUPS = groups


def _run_single_iteration(task: tuple[int, int, float, int, bool]) -> dict[str, object]:
    """Corre un split 80/20 + JL(train only) + HDC + SGD completo.

    Los embeddings Mole-BERT ya estan precalculados una unica vez para todas
    las moleculas antes del loop de iteraciones (no dependen del split), asi
    que cada iteracion solo indexa el embedding matrix y es independiente del
    resto, por lo que puede correr en su propio proceso worker.
    """
    iteration, iteration_seed, test_size, jl_dim, keep_artifacts = task
    assert _WORKER_VALID_ROWS is not None and _WORKER_EMBEDDINGS is not None and _WORKER_LABELS is not None

    process = psutil.Process()
    start_time = time.time()
    cpu_time_start = get_process_cpu_time_seconds(process)

    indices = np.arange(len(_WORKER_VALID_ROWS))
    if _WORKER_GROUPS is None:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=iteration_seed,
            stratify=_WORKER_LABELS,
        )
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=iteration_seed)
        train_idx, test_idx = next(splitter.split(indices, _WORKER_LABELS, groups=_WORKER_GROUPS))
    train_rows = [_WORKER_VALID_ROWS[i] for i in train_idx]
    test_rows = [_WORKER_VALID_ROWS[i] for i in test_idx]
    train_embeddings = _WORKER_EMBEDDINGS[train_idx]
    test_embeddings = _WORKER_EMBEDDINGS[test_idx]
    y_train = _WORKER_LABELS[train_idx]
    y_test = _WORKER_LABELS[test_idx]

    projection_model = fit_projection_model(
        embeddings=train_embeddings,
        jl_dim=jl_dim,
        random_state=iteration_seed,
    )
    train_jl = apply_projection_model(projection_model, train_embeddings)
    test_jl = apply_projection_model(projection_model, test_embeddings)

    train_hdc = convert_hypervectors(np.asarray(train_jl))
    test_hdc = convert_hypervectors(np.asarray(test_jl))

    metrics, y_pred, y_proba, timing = train_and_evaluate_classifier(
        X_train=np.asarray(train_hdc, dtype=np.float32),
        X_test=np.asarray(test_hdc, dtype=np.float32),
        y_train=y_train,
        y_test=y_test,
        random_state=iteration_seed,
    )
    y_score = score_for_predictions_csv(y_proba, y_pred)

    process_metrics = {
        "cpu_time_seconds": max(0.0, get_process_cpu_time_seconds(process) - cpu_time_start),
        "memory_percent": float(process.memory_percent()),
        "memory_mb": float(process.memory_info().rss / 1024 / 1024),
        "elapsed_seconds": float(time.time() - start_time),
    }

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
        for row, true_value, pred_value, score_value in zip(test_rows, y_test, y_pred, y_score)
    ]

    result: dict[str, object] = {
        "iteration": iteration,
        "iteration_seed": iteration_seed,
        "metrics": metrics,
        "timing": timing,
        "process_metrics": process_metrics,
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "prediction_rows": prediction_rows,
    }
    if keep_artifacts:
        result["train_rows"] = train_rows
        result["test_rows"] = test_rows
        result["train_embeddings"] = train_embeddings
        result["test_embeddings"] = test_embeddings
        result["train_jl"] = train_jl
        result["test_jl"] = test_jl
        result["train_hdc"] = train_hdc
        result["test_hdc"] = test_hdc
    return result


def main() -> None:
    overall_start = time.time()
    args = parse_args()
    if not args.checkpoint:
        raise ValueError(
            "Tenes que pasar --checkpoint explicitamente. No deje un checkpoint default implicito."
        )

    if not args.target_column:
        raise ValueError("Tenes que pasar --target-column.")

    outputs = build_output_file_map(args.input_csv, args.output_dir)

    frac_columns = (
        (args.frac_a_column, args.frac_b_column)
        if args.frac_a_column and args.frac_b_column
        else None
    )
    extra_columns = tuple(
        column for column in (args.frac_a_column, args.frac_b_column, args.group_column) if column
    )

    all_rows = load_dataset_rows(
        input_csv=args.input_csv,
        smiles_column=args.smiles_column,
        target_column=args.target_column,
        extra_columns=extra_columns,
    )
    if not all_rows:
        raise ValueError("No quedaron filas validas para ejecutar el pipeline.")

    # Los embeddings Mole-BERT de una molecula no dependen del split 80/20:
    # antes se recalculaban (y se recargaba el checkpoint) por separado para
    # train y test en cada una de las N iteraciones. Se calculan una unica
    # vez ahora y cada iteracion solo indexa el resultado.
    print("Generando embeddings Mole-BERT una unica vez para todas las moleculas validas...")
    embedding_start = time.time()
    valid_rows, embeddings = generate_embeddings_from_rows(
        dataset_rows=all_rows,
        checkpoint=args.checkpoint,
    )
    labels = np.asarray([parse_binary_target(row["target"]) for row in valid_rows], dtype=np.int64)
    if frac_columns:
        fracs = np.asarray(
            [[float(row[frac_columns[0]]), float(row[frac_columns[1]])] for row in valid_rows],
            dtype=np.float32,
        )
        embeddings = np.concatenate([np.asarray(embeddings, dtype=np.float32), fracs], axis=1)
    groups = np.asarray([row[args.group_column] for row in valid_rows]) if args.group_column else None
    print(f"Embeddings generados para {len(valid_rows)} moleculas en {time.time() - embedding_start:.2f}s")

    metrics_tracker = create_metrics_tracker()
    prediction_rows = []

    print_run_configuration(args, len(valid_rows))

    n_jobs = resolve_n_jobs(args.n_jobs, n_rows=len(valid_rows), jl_dim=args.jl_dim)
    print(f"Usando {n_jobs} nucleos para correr {args.iterations} iteraciones en paralelo (JL + HDC + SGD).")

    tasks = [
        (iteration, args.random_state + iteration, args.test_size, args.jl_dim, iteration == 0)
        for iteration in range(args.iterations)
    ]

    if n_jobs > 1 and len(tasks) > 1:
        results = []
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_iteration_worker,
            initargs=(valid_rows, embeddings, labels, groups),
        ) as executor:
            futures = [executor.submit(_run_single_iteration, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                print_iteration_result(done, len(tasks), result)
    else:
        _init_iteration_worker(valid_rows, embeddings, labels, groups)
        results = []
        for done, task in enumerate(tasks, start=1):
            result = _run_single_iteration(task)
            results.append(result)
            print_iteration_result(done, len(tasks), result)

    for result in sorted(results, key=lambda item: item["iteration"]):
        iteration = result["iteration"]
        record_iteration_metrics(
            tracker=metrics_tracker,
            metrics=result["metrics"],
            random_state=result["iteration_seed"],
            process_metrics=result["process_metrics"],
            timing=result["timing"],
            train_size=result["train_size"],
            test_size=result["test_size"],
        )
        prediction_rows.extend(result["prediction_rows"])

        if iteration == 0:
            print_iteration_sizes(result["train_rows"], result["test_rows"])
            write_matrix_csv(outputs["train_embeddings"], result["train_rows"], result["train_embeddings"], prefix="emb")
            write_matrix_csv(outputs["train_jl"], result["train_rows"], np.asarray(result["train_jl"]), prefix="jl")
            write_matrix_csv(outputs["train_hdc"], result["train_rows"], np.asarray(result["train_hdc"]), prefix="hv")
            write_matrix_csv(outputs["test_embeddings"], result["test_rows"], result["test_embeddings"], prefix="emb")
            write_matrix_csv(outputs["test_hdc"], result["test_rows"], np.asarray(result["test_hdc"]), prefix="hv")

    best_idx = int(np.argmax(metrics_tracker["auroc_list"]))
    worst_idx = int(np.argmin(metrics_tracker["auroc_list"]))
    avg_conf_matrix = compute_average_confusion_matrix(metrics_tracker["confusion_matrices"])

    print_final_summary(metrics_tracker, args.iterations)
    timing_summary = {
        "total_wall_clock_seconds": float(time.time() - overall_start),
        "total_training_seconds": float(sum(metrics_tracker["training_seconds_list"])),
        "total_testing_seconds": float(sum(metrics_tracker["testing_seconds_list"])),
    }
    print()
    print("Tiempos totales:")
    print("Tiempo total: {:.2f}s".format(timing_summary["total_wall_clock_seconds"]))
    print("Tiempo total de entrenamiento: {:.2f}s".format(timing_summary["total_training_seconds"]))
    print("Tiempo total de testeo: {:.2f}s".format(timing_summary["total_testing_seconds"]))

    payload = {
        "pipeline": "mole_bert_hdc",
        "dataset_path": args.input_csv,
        "total_rows_original": len(all_rows),
        "total_rows_valid_rdkit": len(valid_rows),
        "train_size": metrics_tracker["train_sizes"][best_idx],
        "test_size": metrics_tracker["test_sizes"][best_idx],
        "smiles_column": args.smiles_column,
        "target_column": args.target_column,
        "random_state_base": args.random_state,
        "test_fraction": args.test_size,
        "iterations": args.iterations,
        "jl_dim": args.jl_dim,
        "best_run": {
            "accuracy": metrics_tracker["accuracy_list"][best_idx],
            "roc_auc": metrics_tracker["auroc_list"][best_idx],
            "balanced_accuracy": metrics_tracker["bacc_list"][best_idx],
            "f1": metrics_tracker["f1_list"][best_idx],
            "precision": metrics_tracker["precision_list"][best_idx],
            "recall": metrics_tracker["recall_list"][best_idx],
            "confusion_matrix": metrics_tracker["confusion_matrices"][best_idx],
            "random_state": metrics_tracker["random_states"][best_idx],
            "cpu_time_seconds": metrics_tracker["cpu_time_seconds_list"][best_idx],
            "memory_percent": metrics_tracker["memory_percent_list"][best_idx],
            "memory_mb": metrics_tracker["memory_mb_list"][best_idx],
            "elapsed_seconds": metrics_tracker["elapsed_seconds_list"][best_idx],
        },
        "worst_run": {
            "accuracy": metrics_tracker["accuracy_list"][worst_idx],
            "roc_auc": metrics_tracker["auroc_list"][worst_idx],
            "balanced_accuracy": metrics_tracker["bacc_list"][worst_idx],
            "f1": metrics_tracker["f1_list"][worst_idx],
            "precision": metrics_tracker["precision_list"][worst_idx],
            "recall": metrics_tracker["recall_list"][worst_idx],
            "confusion_matrix": metrics_tracker["confusion_matrices"][worst_idx],
            "random_state": metrics_tracker["random_states"][worst_idx],
            "cpu_time_seconds": metrics_tracker["cpu_time_seconds_list"][worst_idx],
            "memory_percent": metrics_tracker["memory_percent_list"][worst_idx],
            "memory_mb": metrics_tracker["memory_mb_list"][worst_idx],
            "elapsed_seconds": metrics_tracker["elapsed_seconds_list"][worst_idx],
        },
        "average_metrics": {
            "accuracy": float(np.mean(metrics_tracker["accuracy_list"])),
            "roc_auc": float(np.mean(metrics_tracker["auroc_list"])),
            "balanced_accuracy": float(np.mean(metrics_tracker["bacc_list"])),
            "f1": float(np.mean(metrics_tracker["f1_list"])),
            "precision": float(np.mean(metrics_tracker["precision_list"])),
            "recall": float(np.mean(metrics_tracker["recall_list"])),
            "confusion_matrix": avg_conf_matrix,
        },
        "average_resources": {
            "cpu_time_seconds": float(np.mean(metrics_tracker["cpu_time_seconds_list"])),
            "memory_percent": float(np.mean(metrics_tracker["memory_percent_list"])),
            "memory_mb": float(np.mean(metrics_tracker["memory_mb_list"])),
            "peak_memory_mb": float(np.max(metrics_tracker["memory_mb_list"])),
            "elapsed_seconds": float(np.mean(metrics_tracker["elapsed_seconds_list"])),
        },
        "timing_summary": timing_summary,
    }
    Path(outputs["metrics"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_prediction_rows_csv(outputs["predictions"], prediction_rows)

    print(f"Metricas guardadas en: {outputs['metrics']}")
    print(f"Predicciones guardadas en: {outputs['predictions']}")

if __name__ == "__main__":
    main()
