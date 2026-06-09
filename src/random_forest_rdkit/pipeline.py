from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


DEFAULT_DATASET = "../Mole-BERT/dataset/bbbp/raw/BBBP.csv"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_RANDOM_STATE = 800
DEFAULT_TEST_SIZE = 0.2
DEFAULT_ITERATIONS = 1
FLOAT32_MAX = np.finfo(np.float32).max
DESCRIPTOR_LIST = tuple(Descriptors._descList)
DESCRIPTOR_NAMES = [name for name, _ in DESCRIPTOR_LIST]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena un Random Forest para BBBP usando descriptores de RDKit "
            "y evalua con split 80/20."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Ruta al archivo BBBP.csv.")
    parser.add_argument("--smiles-column", default="smiles", help="Nombre de la columna con SMILES.")
    parser.add_argument("--target-column", default="p_np", help="Nombre de la columna target binaria.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Proporcion usada para test. 0.2 equivale a 80/20.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Seed para split y modelo.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Cantidad de arboles del Random Forest.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Cantidad de corridas 80/20. Default 1 para mantenerlo simple.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta donde se guardan metricas y predicciones.",
    )
    return parser.parse_args()


def resolve_paths(dataset: str, output_dir: str) -> tuple[Path, Path]:
    return Path(dataset).resolve(), Path(output_dir).resolve()


def load_dataset_table(dataset_path: Path, smiles_column: str, target_column: str) -> pd.DataFrame:
    dataframe = pd.read_csv(dataset_path)
    missing_columns = [
        column
        for column in (smiles_column, target_column)
        if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Faltan columnas en el dataset: {missing_columns}. "
            f"Columnas disponibles: {list(dataframe.columns)}"
        )

    dataframe = dataframe[[smiles_column, target_column]].copy()
    dataframe = dataframe.dropna(subset=[smiles_column, target_column])
    dataframe[target_column] = dataframe[target_column].astype(int)
    return dataframe


def smiles_to_descriptor_row(smiles: str) -> list[float] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    values = []
    for _, descriptor_fn in DESCRIPTOR_LIST:
        try:
            values.append(descriptor_fn(mol))
        except Exception:
            values.append(np.nan)
    return values


def sanitize_feature_dataframe(feature_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    numeric_dataframe = feature_dataframe.apply(pd.to_numeric, errors="coerce")

    invalid_before = int(np.isnan(numeric_dataframe.to_numpy(dtype=np.float64)).sum())
    numeric_dataframe = numeric_dataframe.replace([np.inf, -np.inf], np.nan)

    too_large_mask = numeric_dataframe.abs() > FLOAT32_MAX
    too_large_count = int(too_large_mask.to_numpy(dtype=bool).sum())
    numeric_dataframe = numeric_dataframe.mask(too_large_mask, np.nan)

    invalid_after = int(np.isnan(numeric_dataframe.to_numpy(dtype=np.float64)).sum())
    sanitization_info = {
        "num_values_replaced_as_too_large": too_large_count,
        "num_missing_values_before_sanitization": invalid_before,
        "num_missing_values_after_sanitization": invalid_after,
    }
    return numeric_dataframe, sanitization_info


def build_feature_table(dataframe: pd.DataFrame, smiles_column: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    valid_indices = []
    descriptor_rows = []

    for row_index, smiles in dataframe[smiles_column].items():
        descriptors = smiles_to_descriptor_row(smiles)
        if descriptors is None:
            continue
        valid_indices.append(row_index)
        descriptor_rows.append(descriptors)

    if not descriptor_rows:
        raise ValueError("No se pudieron generar descriptores RDKit validos.")

    clean_dataframe = dataframe.loc[valid_indices].reset_index(drop=True)
    feature_dataframe = pd.DataFrame(descriptor_rows, columns=DESCRIPTOR_NAMES)
    feature_dataframe, sanitization_info = sanitize_feature_dataframe(feature_dataframe)
    return clean_dataframe, feature_dataframe, sanitization_info


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float | list[list[int]]]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def average_metric(metric_list: list[float]) -> float:
    return float(sum(metric_list) / len(metric_list))


def average_confusion_matrix(confusion_matrices: list[list[list[int]]]) -> list[list[float]]:
    matrix = np.asarray(confusion_matrices, dtype=float)
    return matrix.mean(axis=0).tolist()


def create_metrics_tracker() -> dict[str, list]:
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


def get_process_cpu_time_seconds(process) -> float:
    if process is None:
        return 0.0
    cpu_times = process.cpu_times()
    return float(cpu_times.user + cpu_times.system)


def measure_resources(process, cpu_time_start: float) -> tuple[float, float, float]:
    if process is None:
        return 0.0, 0.0, 0.0

    cpu_time_seconds = max(0.0, get_process_cpu_time_seconds(process) - cpu_time_start)
    memory_info = process.memory_info()
    memory_percent = float(process.memory_percent())
    memory_mb = float(memory_info.rss / 1024 / 1024)
    return cpu_time_seconds, memory_percent, memory_mb


def build_best_predictions(
    test_rows: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    iteration: int,
    iteration_seed: int,
) -> pd.DataFrame:
    best_predictions = test_rows.copy().reset_index(drop=True)
    best_predictions["y_true"] = y_test
    best_predictions["y_pred"] = y_pred
    best_predictions["y_score"] = y_score
    best_predictions["iteration"] = iteration + 1
    best_predictions["random_state"] = iteration_seed
    return best_predictions


def build_feature_importances(feature_columns, importances) -> pd.DataFrame:
    return pd.DataFrame(
        {"descriptor": feature_columns, "importance": importances}
    ).sort_values("importance", ascending=False)


def run_training_loop(
    X: np.ndarray,
    y: np.ndarray,
    clean_dataframe: pd.DataFrame,
    feature_columns,
    test_size: float,
    random_state: int,
    n_estimators: int,
    iterations: int,
) -> dict[str, object]:
    process = psutil.Process() if psutil is not None else None
    metrics_tracker = create_metrics_tracker()
    detailed_rows = []
    best_predictions = None
    best_feature_importances = None
    best_auroc = float("-inf")
    train_size_used = None
    test_size_used = None

    for iteration in range(iterations):
        iteration_seed = random_state + iteration
        start_time = time.time()
        cpu_time_start = get_process_cpu_time_seconds(process)

        X_train, X_test, y_train, y_test, train_rows, test_rows = train_test_split(
            X,
            y,
            clean_dataframe,
            test_size=test_size,
            random_state=iteration_seed,
            stratify=y,
        )

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=iteration_seed,
            class_weight="balanced",
            n_jobs=-1,
        )
        training_start = time.time()
        model.fit(X_train, y_train)
        training_seconds = time.time() - training_start

        testing_start = time.time()
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_score)
        testing_seconds = time.time() - testing_start

        elapsed_seconds = time.time() - start_time
        cpu_time_seconds, memory_percent, memory_mb = measure_resources(process, cpu_time_start)

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
        metrics_tracker["elapsed_seconds_list"].append(float(elapsed_seconds))
        metrics_tracker["training_seconds_list"].append(float(training_seconds))
        metrics_tracker["testing_seconds_list"].append(float(testing_seconds))

        detailed_rows.append(
            {
                "iteration": iteration + 1,
                "random_state": iteration_seed,
                "accuracy": metrics["accuracy"],
                "roc_auc": metrics["roc_auc"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "cpu_time_seconds": cpu_time_seconds,
                "memory_percent": memory_percent,
                "memory_mb": memory_mb,
                "elapsed_seconds": float(elapsed_seconds),
                "training_seconds": float(training_seconds),
                "testing_seconds": float(testing_seconds),
            }
        )

        if metrics["roc_auc"] > best_auroc:
            best_auroc = metrics["roc_auc"]
            best_predictions = build_best_predictions(
                test_rows=test_rows,
                y_test=y_test,
                y_pred=y_pred,
                y_score=y_score,
                iteration=iteration,
                iteration_seed=iteration_seed,
            )
            best_feature_importances = build_feature_importances(
                feature_columns=feature_columns,
                importances=model.feature_importances_,
            )
            train_size_used = len(y_train)
            test_size_used = len(y_test)

    average_metrics = {
        "accuracy": average_metric(metrics_tracker["accuracy_list"]),
        "roc_auc": average_metric(metrics_tracker["auroc_list"]),
        "balanced_accuracy": average_metric(metrics_tracker["bacc_list"]),
        "f1": average_metric(metrics_tracker["f1_list"]),
        "precision": average_metric(metrics_tracker["precision_list"]),
        "recall": average_metric(metrics_tracker["recall_list"]),
        "confusion_matrix": average_confusion_matrix(metrics_tracker["confusion_matrices"]),
    }

    return {
        "metrics_dict": metrics_tracker,
        "detailed_rows": detailed_rows,
        "best_predictions": best_predictions,
        "best_feature_importances": best_feature_importances,
        "train_size_used": train_size_used,
        "test_size_used": test_size_used,
        "average_metrics": average_metrics,
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def save_outputs(
    output_dir: Path,
    summary: dict,
    detailed_rows: list[dict],
    best_predictions: pd.DataFrame,
    best_feature_importances: pd.DataFrame,
) -> None:
    save_json(output_dir / "metrics.json", summary)
    pd.DataFrame(detailed_rows).to_csv(output_dir / "detailed_results.csv", index=False)
    best_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    best_feature_importances.to_csv(output_dir / "feature_importances.csv", index=False)


def print_dataset_overview(
    dataframe: pd.DataFrame,
    clean_dataframe: pd.DataFrame,
    y: np.ndarray,
    iterations: int,
    test_size: float,
    random_state: int,
) -> None:
    print(f"Dataset original: {len(dataframe)} filas")
    print(f"Dataset valido para RDKit: {len(clean_dataframe)} filas")
    print(f"Clase 0: {int((y == 0).sum())}, Clase 1: {int((y == 1).sum())}")
    print(f"Iteraciones: {iterations}")
    print(
        f"Split por iteracion: {int((1 - test_size) * 100)}/{int(test_size * 100)} "
        f"con seed {random_state} + iteracion"
    )


def print_feature_sanitization_summary(sanitization_info: dict[str, int]) -> None:
    print(
        "Valores no finitos o demasiado grandes convertidos a NaN: "
        f"{sanitization_info['num_values_replaced_as_too_large']}"
    )
    print(
        "Valores faltantes antes de sanitizar: "
        f"{sanitization_info['num_missing_values_before_sanitization']}"
    )
    print(
        "Valores faltantes despues de sanitizar: "
        f"{sanitization_info['num_missing_values_after_sanitization']}"
    )


def print_iteration_style_report(metrics_dict: dict[str, list], iterations: int) -> None:
    max_auroc = max(metrics_dict["auroc_list"])
    max_auroc_idx = metrics_dict["auroc_list"].index(max_auroc)
    min_auroc = min(metrics_dict["auroc_list"])
    min_auroc_idx = metrics_dict["auroc_list"].index(min_auroc)

    print()
    print("Stats corresponding to Maximum AUROC are:")
    print("Accuracy: ", metrics_dict["accuracy_list"][max_auroc_idx])
    print("Auroc: ", metrics_dict["auroc_list"][max_auroc_idx])
    print("Bacc: ", metrics_dict["bacc_list"][max_auroc_idx])
    print("F1: ", metrics_dict["f1_list"][max_auroc_idx])
    print("Precision: ", metrics_dict["precision_list"][max_auroc_idx])
    print("Recall: ", metrics_dict["recall_list"][max_auroc_idx])
    print("Confusion Matrix: ", metrics_dict["confusion_matrices"][max_auroc_idx])
    print("Random State: ", metrics_dict["random_states"][max_auroc_idx])
    print("CPU Total Time: {:.2f}s".format(metrics_dict["cpu_time_seconds_list"][max_auroc_idx]))
    print(
        "Memory Usage: {:.2f}% ({:.2f} MB)".format(
            metrics_dict["memory_percent_list"][max_auroc_idx],
            metrics_dict["memory_mb_list"][max_auroc_idx],
        )
    )
    print("Elapsed Seconds: {:.2f}".format(metrics_dict["elapsed_seconds_list"][max_auroc_idx]))

    print()
    print("Stats corresponding to Minimum AUROC are:")
    print("Accuracy: ", metrics_dict["accuracy_list"][min_auroc_idx])
    print("Auroc: ", metrics_dict["auroc_list"][min_auroc_idx])
    print("Bacc: ", metrics_dict["bacc_list"][min_auroc_idx])
    print("F1: ", metrics_dict["f1_list"][min_auroc_idx])
    print("Precision: ", metrics_dict["precision_list"][min_auroc_idx])
    print("Recall: ", metrics_dict["recall_list"][min_auroc_idx])
    print("Confusion Matrix: ", metrics_dict["confusion_matrices"][min_auroc_idx])
    print("Random State: ", metrics_dict["random_states"][min_auroc_idx])
    print("CPU Total Time: {:.2f}s".format(metrics_dict["cpu_time_seconds_list"][min_auroc_idx]))
    print(
        "Memory Usage: {:.2f}% ({:.2f} MB)".format(
            metrics_dict["memory_percent_list"][min_auroc_idx],
            metrics_dict["memory_mb_list"][min_auroc_idx],
        )
    )
    print("Elapsed Seconds: {:.2f}".format(metrics_dict["elapsed_seconds_list"][min_auroc_idx]))

    print()
    print(f"Average Stats for {iterations} iterations")
    print("Accuracy: ", average_metric(metrics_dict["accuracy_list"]))
    print("Auroc: ", average_metric(metrics_dict["auroc_list"]))
    print("Bacc: ", average_metric(metrics_dict["bacc_list"]))
    print("F1: ", average_metric(metrics_dict["f1_list"]))
    print("Precision: ", average_metric(metrics_dict["precision_list"]))
    print("Recall: ", average_metric(metrics_dict["recall_list"]))
    print("Average Confusion Matrix: ", average_confusion_matrix(metrics_dict["confusion_matrices"]))
    print("Average CPU Total Time: {:.2f}s".format(average_metric(metrics_dict["cpu_time_seconds_list"])))
    print(
        "Average Memory Usage: {:.2f}% ({:.2f} MB)".format(
            average_metric(metrics_dict["memory_percent_list"]),
            average_metric(metrics_dict["memory_mb_list"]),
        )
    )
    print("Peak Memory Usage: {:.2f} MB".format(max(metrics_dict["memory_mb_list"])))
    print("Average Elapsed Seconds: {:.2f}".format(average_metric(metrics_dict["elapsed_seconds_list"])))


def print_save_summary(output_dir: Path) -> None:
    print()
    print("Saving performance metrics and best iteration outputs...")
    print(f"Metricas guardadas en: {output_dir / 'metrics.json'}")
    print(f"Resultados detallados guardados en: {output_dir / 'detailed_results.csv'}")
    print(f"Predicciones guardadas en: {output_dir / 'test_predictions.csv'}")
    print(f"Importancias guardadas en: {output_dir / 'feature_importances.csv'}")


def main() -> None:
    overall_start = time.time()
    args = parse_args()
    dataset_path, output_dir = resolve_paths(args.dataset, args.output_dir)

    dataframe = load_dataset_table(
        dataset_path=dataset_path,
        smiles_column=args.smiles_column,
        target_column=args.target_column,
    )
    clean_dataframe, feature_dataframe, sanitization_info = build_feature_table(
        dataframe=dataframe,
        smiles_column=args.smiles_column,
    )

    X = feature_dataframe.to_numpy(dtype=float)
    y = clean_dataframe[args.target_column].to_numpy(dtype=int)

    print_dataset_overview(
        dataframe=dataframe,
        clean_dataframe=clean_dataframe,
        y=y,
        iterations=args.iterations,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print_feature_sanitization_summary(sanitization_info)

    training_result = run_training_loop(
        X=X,
        y=y,
        clean_dataframe=clean_dataframe,
        feature_columns=feature_dataframe.columns,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        iterations=args.iterations,
    )

    summary = {
        "dataset_path": str(dataset_path),
        "total_rows_original": int(len(dataframe)),
        "total_rows_valid_rdkit": int(len(clean_dataframe)),
        "train_size": int(training_result["train_size_used"]),
        "test_size": int(training_result["test_size_used"]),
        "smiles_column": args.smiles_column,
        "target_column": args.target_column,
        "random_state_base": args.random_state,
        "test_fraction": args.test_size,
        "n_estimators": args.n_estimators,
        "iterations": args.iterations,
        "num_descriptors": int(feature_dataframe.shape[1]),
        "feature_sanitization": sanitization_info,
        "metrics_by_iteration": training_result["metrics_dict"],
        "average_metrics": training_result["average_metrics"],
        "timing_summary": {
            "total_wall_clock_seconds": float(time.time() - overall_start),
            "total_training_seconds": float(sum(training_result["metrics_dict"]["training_seconds_list"])),
            "total_testing_seconds": float(sum(training_result["metrics_dict"]["testing_seconds_list"])),
        },
    }

    save_outputs(
        output_dir=output_dir,
        summary=summary,
        detailed_rows=training_result["detailed_rows"],
        best_predictions=training_result["best_predictions"],
        best_feature_importances=training_result["best_feature_importances"],
    )

    print_iteration_style_report(training_result["metrics_dict"], args.iterations)
    print()
    print("Tiempos totales:")
    print("Tiempo total:", f"{summary['timing_summary']['total_wall_clock_seconds']:.2f}s")
    print("Tiempo total de entrenamiento:", f"{summary['timing_summary']['total_training_seconds']:.2f}s")
    print("Tiempo total de testeo:", f"{summary['timing_summary']['total_testing_seconds']:.2f}s")
    print_save_summary(output_dir)


if __name__ == "__main__":
    main()
