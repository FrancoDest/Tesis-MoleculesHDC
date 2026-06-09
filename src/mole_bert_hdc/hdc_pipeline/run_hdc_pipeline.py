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
import json
import time
from pathlib import Path

import numpy as np
import psutil
from sklearn.model_selection import train_test_split

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
    parser.add_argument("--target-column", required=True, help="Columna target binaria.")
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
    return parser.parse_args()


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
        "cpu_time_seconds_list": [],
        "memory_percent_list": [],
        "memory_mb_list": [],
        "elapsed_seconds_list": [],
        "training_seconds_list": [],
        "testing_seconds_list": [],
    }


def print_run_configuration(args: argparse.Namespace, total_rows: int) -> None:
    print(f"Filas validas: {total_rows}")
    print(f"Iteraciones: {args.iterations}")
    print(f"Random state base: {args.random_state}")
    print(f"Split por iteracion: {int((1 - args.test_size) * 100)}/{int(args.test_size * 100)} con seed base + iteracion")
    print("JL: se ajusta solo con train y se aplica al test con el mismo proyector")


def print_iteration_sizes(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]]) -> None:
    print(f"Train size (80% aprox): {len(train_rows)}")
    print(f"Test size (20% aprox): {len(test_rows)}")


def collect_process_metrics(
    process: psutil.Process,
    cpu_time_start: float,
    iteration_start: float,
) -> dict[str, float]:
    memory_info = process.memory_info()
    return {
        "cpu_time_seconds": max(0.0, get_process_cpu_time_seconds(process) - cpu_time_start),
        "memory_percent": float(process.memory_percent()),
        "memory_mb": float(memory_info.rss / 1024 / 1024),
        "elapsed_seconds": float(time.time() - iteration_start),
    }


def record_iteration_metrics(
    tracker: dict[str, list[object]],
    metrics: dict[str, object],
    random_state: int,
    process_metrics: dict[str, float],
    timing: dict[str, float],
) -> None:
    tracker["accuracy_list"].append(metrics["Accuracy"])
    tracker["auroc_list"].append(metrics["AUROC"])
    tracker["bacc_list"].append(metrics["Bacc"])
    tracker["f1_list"].append(metrics["F1"])
    tracker["precision_list"].append(metrics["Precision"])
    tracker["recall_list"].append(metrics["Recall"])
    tracker["confusion_matrices"].append(metrics["ConfusionMatrix"])
    tracker["random_states"].append(random_state)
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
    print("Confusion Matrix promedio:", average_confusion_matrix)
    print("Average CPU Total Time: {:.2f}s".format(float(np.mean(metrics_tracker["cpu_time_seconds_list"]))))
    print(
        "Average Memory Usage: {:.2f}% ({:.2f} MB)".format(
            float(np.mean(metrics_tracker["memory_percent_list"])),
            float(np.mean(metrics_tracker["memory_mb_list"])),
        )
    )
    print("Peak Memory Usage: {:.2f} MB".format(float(np.max(metrics_tracker["memory_mb_list"]))))


def main() -> None:
    overall_start = time.time()
    args = parse_args()
    if not args.checkpoint:
        raise ValueError(
            "Tenes que pasar --checkpoint explicitamente. No deje un checkpoint default implicito."
        )

    outputs = build_output_file_map(args.input_csv, args.output_dir)
    process = psutil.Process()

    all_rows = load_dataset_rows(
        input_csv=args.input_csv,
        smiles_column=args.smiles_column,
        target_column=args.target_column,
    )
    if not all_rows:
        raise ValueError("No quedaron filas validas para ejecutar el pipeline.")

    metrics_tracker = create_metrics_tracker()
    prediction_rows = []

    first_iteration_saved = False

    print_run_configuration(args, len(all_rows))

    for iteration in range(args.iterations):
        start_time = time.time()
        cpu_time_start = get_process_cpu_time_seconds(process)
        iteration_seed = args.random_state + iteration

        train_rows, test_rows = train_test_split(
            all_rows,
            test_size=args.test_size,
            random_state=iteration_seed,
            stratify=[parse_binary_target(row["target"]) for row in all_rows],
        )

        valid_train_rows, train_embeddings = generate_embeddings_from_rows(
            dataset_rows=train_rows,
            checkpoint=args.checkpoint,
        )
        valid_test_rows, test_embeddings = generate_embeddings_from_rows(
            dataset_rows=test_rows,
            checkpoint=args.checkpoint,
        )

        projection_model = fit_projection_model(
            embeddings=train_embeddings,
            jl_dim=args.jl_dim,
            random_state=iteration_seed,
        )
        train_jl = apply_projection_model(projection_model, train_embeddings)
        test_jl = apply_projection_model(projection_model, test_embeddings)

        train_hdc = convert_hypervectors(np.asarray(train_jl))
        test_hdc = convert_hypervectors(np.asarray(test_jl))

        y_train = np.asarray([parse_binary_target(row["target"]) for row in valid_train_rows], dtype=np.int64)
        y_test = np.asarray([parse_binary_target(row["target"]) for row in valid_test_rows], dtype=np.int64)

        metrics, y_pred, y_score, timing = train_and_evaluate_classifier(
            X_train=np.asarray(train_hdc, dtype=np.float32),
            X_test=np.asarray(test_hdc, dtype=np.float32),
            y_train=y_train,
            y_test=y_test,
            random_state=iteration_seed,
        )

        process_metrics = collect_process_metrics(process, cpu_time_start, start_time)
        record_iteration_metrics(
            tracker=metrics_tracker,
            metrics=metrics,
            random_state=iteration_seed,
            process_metrics=process_metrics,
            timing=timing,
        )

        if iteration == 0:
            print_iteration_sizes(valid_train_rows, valid_test_rows)

        if not first_iteration_saved:
            write_matrix_csv(outputs["train_embeddings"], valid_train_rows, train_embeddings, prefix="emb")
            write_matrix_csv(outputs["train_jl"], valid_train_rows, np.asarray(train_jl), prefix="jl")
            write_matrix_csv(outputs["train_hdc"], valid_train_rows, np.asarray(train_hdc), prefix="hv")
            write_matrix_csv(outputs["test_embeddings"], valid_test_rows, test_embeddings, prefix="emb")
            write_matrix_csv(outputs["test_hdc"], valid_test_rows, np.asarray(test_hdc), prefix="hv")
            first_iteration_saved = True

        for row, true_value, pred_value, score_value in zip(valid_test_rows, y_test, y_pred, y_score):
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
        "random_state": args.random_state,
        "iterations": args.iterations,
        "test_size": args.test_size,
        "jl_dim": args.jl_dim,
        "train_size": int(round(len(all_rows) * (1.0 - args.test_size))),
        "eval_size": int(len(all_rows) - round(len(all_rows) * (1.0 - args.test_size))),
        "best_run": {
            "Accuracy": metrics_tracker["accuracy_list"][best_idx],
            "AUROC": metrics_tracker["auroc_list"][best_idx],
            "Bacc": metrics_tracker["bacc_list"][best_idx],
            "F1": metrics_tracker["f1_list"][best_idx],
            "Precision": metrics_tracker["precision_list"][best_idx],
            "Recall": metrics_tracker["recall_list"][best_idx],
            "ConfusionMatrix": metrics_tracker["confusion_matrices"][best_idx],
            "RandomState": metrics_tracker["random_states"][best_idx],
        },
        "worst_run": {
            "Accuracy": metrics_tracker["accuracy_list"][worst_idx],
            "AUROC": metrics_tracker["auroc_list"][worst_idx],
            "Bacc": metrics_tracker["bacc_list"][worst_idx],
            "F1": metrics_tracker["f1_list"][worst_idx],
            "Precision": metrics_tracker["precision_list"][worst_idx],
            "Recall": metrics_tracker["recall_list"][worst_idx],
            "ConfusionMatrix": metrics_tracker["confusion_matrices"][worst_idx],
            "RandomState": metrics_tracker["random_states"][worst_idx],
        },
        "average_metrics": {
            "Accuracy": float(np.mean(metrics_tracker["accuracy_list"])),
            "AUROC": float(np.mean(metrics_tracker["auroc_list"])),
            "Bacc": float(np.mean(metrics_tracker["bacc_list"])),
            "F1": float(np.mean(metrics_tracker["f1_list"])),
            "Precision": float(np.mean(metrics_tracker["precision_list"])),
            "Recall": float(np.mean(metrics_tracker["recall_list"])),
            "ConfusionMatrix": avg_conf_matrix,
        },
        "average_resources": {
            "CPU_Total_Time_Seconds": float(np.mean(metrics_tracker["cpu_time_seconds_list"])),
            "Memory_Usage_Percent": float(np.mean(metrics_tracker["memory_percent_list"])),
            "Memory_MB": float(np.mean(metrics_tracker["memory_mb_list"])),
            "Peak_Memory_MB": float(np.max(metrics_tracker["memory_mb_list"])),
            "Elapsed_Seconds": float(np.mean(metrics_tracker["elapsed_seconds_list"])),
        },
        "timing_summary": timing_summary,
    }
    Path(outputs["metrics"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_prediction_rows_csv(outputs["predictions"], prediction_rows)

    print(f"Metricas guardadas en: {outputs['metrics']}")
    print(f"Predicciones guardadas en: {outputs['predictions']}")

if __name__ == "__main__":
    main()
