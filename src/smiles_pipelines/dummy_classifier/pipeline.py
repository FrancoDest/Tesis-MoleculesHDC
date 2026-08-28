from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


RDLogger.DisableLog("rdApp.*")

DEFAULT_DATASET = "../Mole-BERT/dataset/bbbp/raw/BBBP.csv"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_RANDOM_STATE = 800
DEFAULT_TEST_SIZE = 0.2
DEFAULT_ITERATIONS = 1
DEFAULT_STRATEGY = "most_frequent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline dummy (sklearn DummyClassifier) para datasets binarios/multiclase "
            "tipo BBBP/MUTAG: ignora por completo las moleculas y solo mira la "
            "distribucion de clases del train, para tener un piso de comparacion "
            "contra el resto de los pipelines."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Ruta al archivo BBBP.csv.")
    parser.add_argument("--smiles-column", default="smiles", help="Nombre de la columna con SMILES.")
    parser.add_argument("--target-column", default="p_np", help="Nombre de la columna target.")
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
        help="Seed para el split y para el DummyClassifier (usada solo por 'stratified'/'uniform').",
    )
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        choices=["most_frequent", "stratified", "uniform"],
        help=(
            "Estrategia del DummyClassifier: 'most_frequent' siempre predice la clase "
            "mayoritaria (baseline estandar de accuracy); 'stratified' predice al azar "
            "respetando la proporcion de clases del train; 'uniform' predice al azar "
            "con probabilidad uniforme entre clases."
        ),
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
    parser.add_argument(
        "--frac-a-column",
        default=None,
        help="Columna opcional de composicion . Se conserva en las salidas pero no se usa como feature: el dummy ignora todas las features.",
    )
    parser.add_argument("--frac-b-column", default=None, help="Ver --frac-a-column.")
    parser.add_argument(
        "--group-column",
        default=None,
        help=(
            "Columna opcional para agrupar el split train/test: "
            "garantiza que un mismo grupo no quede repartido entre train y test. Si no "
            "se pasa, el split es aleatorio estratificado por clase como siempre."
        ),
    )
    return parser.parse_args()


def resolve_paths(dataset: str, output_dir: str) -> tuple[Path, Path]:
    return Path(dataset).resolve(), Path(output_dir).resolve()


def load_dataset_table(
    dataset_path: Path, smiles_column: str, target_column: str, extra_columns: tuple[str, ...] = ()
) -> pd.DataFrame:
    dataframe = pd.read_csv(dataset_path)
    required = [smiles_column, target_column, *extra_columns]
    missing_columns = [column for column in required if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            f"Faltan columnas en el dataset: {missing_columns}. "
            f"Columnas disponibles: {list(dataframe.columns)}"
        )

    dataframe = dataframe[required].copy()
    dataframe = dataframe.dropna(subset=[smiles_column, target_column])
    dataframe[target_column] = dataframe[target_column].astype(int)
    return dataframe.reset_index(drop=True)


def filter_rdkit_valid(dataframe: pd.DataFrame, smiles_column: str) -> pd.DataFrame:
    """Descarta las filas cuyo SMILES RDKit no puede parsear.

    El dummy ignora la molecula para predecir, pero si no aplica el mismo
    filtro que el resto termina evaluando sobre un conjunto distinto: en BBBP
    son 2050 moleculas contra las 2039 que usan los baselines con descriptores
    (random_forest_rdkit, logistic_regression_rdkit, nn_classifier_rdkit), y
    el piso de comparacion deja de ser comparable. El criterio es el mismo que
    usa `smiles_to_descriptor_row` en esos pipelines: `Chem.MolFromSmiles`
    devuelve None -> la fila se cae."""
    valid_mask = [Chem.MolFromSmiles(str(smiles)) is not None for smiles in dataframe[smiles_column]]
    return dataframe.loc[valid_mask].reset_index(drop=True)


def score_for_predictions_csv(y_proba: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Probabilidad a guardar en test_predictions.csv: para binario, la de la
    clase positiva; para multiclase, la de la clase predicha."""
    if y_proba.shape[1] == 2:
        return y_proba[:, 1]
    return y_proba[np.arange(len(y_pred)), y_pred]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float | list[list[int]]]:
    is_multiclass = y_proba.shape[1] > 2
    average = "macro" if is_multiclass else "binary"
    metrics: dict[str, float | list[list[int]]] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        if is_multiclass:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        else:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
    except ValueError:
        metrics["roc_auc"] = 0.5
    return metrics


def average_metric(metric_list: list[float]) -> float:
    return float(sum(metric_list) / len(metric_list))


def std_metric(metric_list: list[float]) -> float:
    """Desviacion estandar muestral (ddof=1). Con 1 sola iteracion no hay
    variabilidad que medir, asi que devuelve 0.0 en vez de dividir por 0."""
    if len(metric_list) < 2:
        return 0.0
    mean = average_metric(metric_list)
    variance = sum((value - mean) ** 2 for value in metric_list) / (len(metric_list) - 1)
    return float(variance ** 0.5)


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


def run_snapshot_from_tracker(tracker: dict[str, list], idx: int) -> dict[str, object]:
    return {
        "accuracy": tracker["accuracy_list"][idx],
        "roc_auc": tracker["auroc_list"][idx],
        "balanced_accuracy": tracker["bacc_list"][idx],
        "f1": tracker["f1_list"][idx],
        "precision": tracker["precision_list"][idx],
        "recall": tracker["recall_list"][idx],
        "confusion_matrix": tracker["confusion_matrices"][idx],
        "random_state": tracker["random_states"][idx],
        "cpu_time_seconds": tracker["cpu_time_seconds_list"][idx],
        "memory_percent": tracker["memory_percent_list"][idx],
        "memory_mb": tracker["memory_mb_list"][idx],
        "elapsed_seconds": tracker["elapsed_seconds_list"][idx],
    }


def average_resources_from_tracker(tracker: dict[str, list]) -> dict[str, float]:
    return {
        "cpu_time_seconds": average_metric(tracker["cpu_time_seconds_list"]),
        "memory_percent": average_metric(tracker["memory_percent_list"]),
        "memory_mb": average_metric(tracker["memory_mb_list"]),
        "peak_memory_mb": max(tracker["memory_mb_list"]) if tracker["memory_mb_list"] else 0.0,
        "elapsed_seconds": average_metric(tracker["elapsed_seconds_list"]),
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


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    clean_dataframe: pd.DataFrame,
    test_size: float,
    iteration_seed: int,
    groups: np.ndarray | None,
):
    """Split aleatorio estratificado por clase de siempre, salvo que se pase
    `groups` : ahi se usa GroupShuffleSplit para que un
    mismo grupo (par de monomeros) no quede repartido entre train y test."""
    if groups is None:
        return train_test_split(X, y, clean_dataframe, test_size=test_size, random_state=iteration_seed, stratify=y)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=iteration_seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
        clean_dataframe.iloc[train_idx],
        clean_dataframe.iloc[test_idx],
    )


def run_training_loop(
    X: np.ndarray,
    y: np.ndarray,
    clean_dataframe: pd.DataFrame,
    test_size: float,
    random_state: int,
    strategy: str,
    iterations: int,
    groups: np.ndarray | None = None,
) -> dict[str, object]:
    process = psutil.Process() if psutil is not None else None
    metrics_tracker = create_metrics_tracker()
    detailed_rows = []
    best_predictions = None
    best_auroc = float("-inf")
    train_size_used = None
    test_size_used = None

    for iteration in range(iterations):
        iteration_seed = random_state + iteration
        start_time = time.time()
        cpu_time_start = get_process_cpu_time_seconds(process)

        X_train, X_test, y_train, y_test, train_rows, test_rows = split_train_test(
            X, y, clean_dataframe, test_size=test_size, iteration_seed=iteration_seed, groups=groups
        )

        model = DummyClassifier(strategy=strategy, random_state=iteration_seed)
        training_start = time.time()
        model.fit(X_train, y_train)
        training_seconds = time.time() - training_start

        testing_start = time.time()
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba)
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
                y_score=score_for_predictions_csv(y_proba, y_pred),
                iteration=iteration,
                iteration_seed=iteration_seed,
            )
            train_size_used = len(y_train)
            test_size_used = len(y_test)

        print(
            f"[{iteration + 1}/{iterations}] accuracy={metrics['accuracy']:.4f} "
            f"auroc={metrics['roc_auc']:.4f} balanced_accuracy={metrics['balanced_accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
            f"random_state={iteration_seed} cpu_time={cpu_time_seconds:.2f}s "
            f"memory={memory_mb:.1f}MB elapsed={elapsed_seconds:.1f}s"
        )

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
) -> None:
    save_json(output_dir / "metrics.json", summary)
    pd.DataFrame(detailed_rows).to_csv(output_dir / "detailed_results.csv", index=False)
    best_predictions.to_csv(output_dir / "test_predictions.csv", index=False)


def print_dataset_overview(
    dataframe: pd.DataFrame,
    clean_dataframe: pd.DataFrame,
    y: np.ndarray,
    iterations: int,
    test_size: float,
    random_state: int,
    strategy: str,
) -> None:
    print(f"Dataset original: {len(dataframe)} filas")
    print(f"Dataset valido para RDKit: {len(clean_dataframe)} filas")
    class_counts = ", ".join(f"Clase {label}: {int((y == label).sum())}" for label in sorted(set(y.tolist())))
    print(class_counts)
    print(f"Estrategia: {strategy}")
    print(f"Iteraciones: {iterations}")
    print(
        f"Split por iteracion: {int((1 - test_size) * 100)}/{int(test_size * 100)} "
        f"con seed {random_state} + iteracion"
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

    print()
    print(f"Average Stats for {iterations} iterations")
    print("Accuracy: ", average_metric(metrics_dict["accuracy_list"]))
    print("Auroc: ", average_metric(metrics_dict["auroc_list"]))
    print("Bacc: ", average_metric(metrics_dict["bacc_list"]))
    print("F1: ", average_metric(metrics_dict["f1_list"]))
    print("Precision: ", average_metric(metrics_dict["precision_list"]))
    print("Recall: ", average_metric(metrics_dict["recall_list"]))
    print()
    print(f"Standard Deviation for {iterations} iterations")
    print("Accuracy: ", std_metric(metrics_dict["accuracy_list"]))
    print("Auroc: ", std_metric(metrics_dict["auroc_list"]))
    print("Bacc: ", std_metric(metrics_dict["bacc_list"]))
    print("F1: ", std_metric(metrics_dict["f1_list"]))
    print("Precision: ", std_metric(metrics_dict["precision_list"]))
    print("Recall: ", std_metric(metrics_dict["recall_list"]))
    print("Average Confusion Matrix: ", average_confusion_matrix(metrics_dict["confusion_matrices"]))


def print_save_summary(output_dir: Path) -> None:
    print()
    print("Saving performance metrics and best iteration outputs...")
    print(f"Metricas guardadas en: {output_dir / 'metrics.json'}")
    print(f"Resultados detallados guardados en: {output_dir / 'detailed_results.csv'}")
    print(f"Predicciones guardadas en: {output_dir / 'test_predictions.csv'}")


def main() -> None:
    overall_start = time.time()
    args = parse_args()
    dataset_path, output_dir = resolve_paths(args.dataset, args.output_dir)

    extra_columns = tuple(
        column for column in (args.frac_a_column, args.frac_b_column, args.group_column) if column
    )
    dataframe = load_dataset_table(
        dataset_path=dataset_path,
        smiles_column=args.smiles_column,
        target_column=args.target_column,
        extra_columns=extra_columns,
    )

    clean_dataframe = filter_rdkit_valid(dataframe, args.smiles_column)
    if clean_dataframe.empty:
        raise ValueError("No quedaron moleculas con SMILES valido para RDKit.")

    y = clean_dataframe[args.target_column].to_numpy(dtype=int)
    X = np.zeros((len(y), 1))
    groups = clean_dataframe[args.group_column].to_numpy() if args.group_column else None

    print_dataset_overview(
        dataframe=dataframe,
        clean_dataframe=clean_dataframe,
        y=y,
        iterations=args.iterations,
        test_size=args.test_size,
        random_state=args.random_state,
        strategy=args.strategy,
    )

    training_result = run_training_loop(
        X=X,
        y=y,
        clean_dataframe=clean_dataframe,
        test_size=args.test_size,
        random_state=args.random_state,
        strategy=args.strategy,
        iterations=args.iterations,
        groups=groups,
    )

    summary = {
        "pipeline": "dummy_classifier",
        "dataset_path": str(dataset_path),
        "total_rows_original": int(len(dataframe)),
        "total_rows_valid_rdkit": int(len(clean_dataframe)),
        "train_size": int(training_result["train_size_used"]),
        "test_size": int(training_result["test_size_used"]),
        "smiles_column": args.smiles_column,
        "target_column": args.target_column,
        "random_state_base": args.random_state,
        "test_fraction": args.test_size,
        "strategy": args.strategy,
        "iterations": args.iterations,
        "metrics_by_iteration": training_result["metrics_dict"],
        "average_metrics": training_result["average_metrics"],
        "average_resources": average_resources_from_tracker(training_result["metrics_dict"]),
        "best_run": run_snapshot_from_tracker(
            training_result["metrics_dict"],
            training_result["metrics_dict"]["auroc_list"].index(max(training_result["metrics_dict"]["auroc_list"])),
        ),
        "worst_run": run_snapshot_from_tracker(
            training_result["metrics_dict"],
            training_result["metrics_dict"]["auroc_list"].index(min(training_result["metrics_dict"]["auroc_list"])),
        ),
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
