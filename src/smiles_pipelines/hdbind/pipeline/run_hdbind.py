"""HDBind: encoding de fingerprints ECFP como hipervectores hiperdimensionales
mediante proyeccion aleatoria fija, clasificados con una memoria asociativa
entrenada por reentrenamiento tipo perceptron (single-pass bundling +
correccion iterativa de errores), en vez del bundling puro de un solo paso
que usan graphHD/MoleHD en este repo.

Fiel al metodo de LLNL/hdbind (https://github.com/LLNL/hdbind), variante
RP-ECFP (RPEncoder + train_hdc de hdpy/model.py):
  - fingerprint ECFP binario -> bipolar {-1,+1}
  - hv = sign(fingerprint_bipolar @ proyeccion_aleatoria_bipolar)
  - memoria asociativa: 1 vector por clase, suma de los hv de entrenamiento
  - reentrenamiento: por epoca, para cada ejemplo mal clasificado, se suma su
    hv a la clase correcta y se resta de la clase predicha (equivocada)
  - prediccion: similitud coseno contra cada vector de clase, argmax

Notas de diseno pensadas a partir de dos bugs reales encontrados en graphHD
en este mismo repo:
  - Se usa similitud COSENO (no Hamming/igualdad exacta) para que la
    comparacion sea invariante a la escala de los vectores acumulados (un
    vector de clase que sumo miles de muestras no rompe la comparacion).
  - El fingerprint/proyeccion se calcula una unica vez para todas las
    moleculas (no por iteracion), y el computo de fingerprints con RDKit via
    ProcessPoolExecutor se hace ANTES de tocar torch: asi el fork no hereda
    ningun estado de threads de torch (la causa del deadlock que encontramos
    en graphHD al forkear despues de haber usado torch en el proceso padre).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
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

DEFAULT_DIM = 10048
DEFAULT_ECFP_RADIUS = 2
DEFAULT_ECFP_BITS = 1024
DEFAULT_NUM_EPOCHS = 10
DEFAULT_ITERATIONS = 100
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 800

# Filas por lote al proyectar/puntuar hipervectores. Acota el tamano de los
# tensores float32 intermedios sin importar cuantas moleculas tenga el
# dataset: 4096 x 10048 float32 son ~165 MB, contra los 1,65 GB que ocupa
# HIV entero (41127 moleculas). No cambia el resultado, solo el pico de RAM.
HV_BATCH_ROWS = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "HDBind: fingerprints ECFP proyectados a hipervectores hiperdimensionales, "
            "clasificados con memoria asociativa + reentrenamiento tipo perceptron."
        )
    )
    parser.add_argument("--dataset", required=True, help="CSV con columnas de smiles y target.")
    parser.add_argument("--smiles-column", default="smiles", help="Nombre de la columna de SMILES.")
    parser.add_argument("--target-column", default="label", help="Nombre de la columna target binaria.")
    parser.add_argument("--id-column", default="id", help="Nombre de la columna id (para las predicciones).")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM, help="Dimension de los hipervectores.")
    parser.add_argument("--ecfp-radius", type=int, default=DEFAULT_ECFP_RADIUS, help="Radio del fingerprint ECFP (Morgan).")
    parser.add_argument("--ecfp-bits", type=int, default=DEFAULT_ECFP_BITS, help="Longitud del fingerprint ECFP.")
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help="Epocas de reentrenamiento tipo perceptron sobre la memoria asociativa.",
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Cantidad de splits 80/20.")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Fraccion de test.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Semilla base.")
    parser.add_argument("--output-dir", default="/run_outputs/artifacts", help="Directorio de salida.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Nucleos para calcular fingerprints ECFP con RDKit. 0 (default) usa todos los disponibles.",
    )
    parser.add_argument(
        "--frac-a-column",
        default=None,
        help=(
            "Columna opcional en --dataset con una fraccion de composicion "
            ". Si se pasa junto con --frac-b-column, se suma "
            "como feature numerica extra al fingerprint antes de la proyeccion aleatoria."
        ),
    )
    parser.add_argument("--frac-b-column", default=None, help="Ver --frac-a-column.")
    parser.add_argument(
        "--group-column",
        default=None,
        help=(
            "Columna opcional en --dataset para agrupar el split train/test "
            ": garantiza que un mismo grupo no quede "
            "repartido entre train y test."
        ),
    )
    return parser.parse_args()


def resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs > 0:
        return n_jobs
    return os.cpu_count() or 1


def compute_ecfp_fingerprint(args: tuple[str, int, int]) -> np.ndarray | None:
    smiles, radius, n_bits = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.frombuffer(bit_vect.ToBitString().encode("ascii"), dtype=np.uint8) - ord("0")


def load_dataset_table(
    dataset_path: Path,
    smiles_column: str,
    target_column: str,
    id_column: str,
    extra_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    dataframe = pd.read_csv(dataset_path)
    missing_columns = [
        column for column in (smiles_column, target_column, *extra_columns) if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Faltan columnas en el dataset: {missing_columns}. Columnas disponibles: {list(dataframe.columns)}"
        )
    if id_column not in dataframe.columns:
        dataframe = dataframe.copy()
        dataframe[id_column] = np.arange(len(dataframe))

    columns_to_keep = [id_column, smiles_column, target_column, *extra_columns]
    dataframe = dataframe[columns_to_keep].copy()
    dataframe = dataframe.dropna(subset=[smiles_column, target_column])
    dataframe[target_column] = dataframe[target_column].astype(int)
    return dataframe.reset_index(drop=True)


def build_fingerprint_table(
    dataframe: pd.DataFrame, smiles_column: str, radius: int, n_bits: int, n_jobs: int
) -> tuple[pd.DataFrame, np.ndarray]:
    smiles_values = dataframe[smiles_column].tolist()
    tasks = [(smiles, radius, n_bits) for smiles in smiles_values]

    if n_jobs > 1 and len(tasks) > 1:
        chunksize = max(1, len(tasks) // (n_jobs * 4))
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            fingerprints = list(executor.map(compute_ecfp_fingerprint, tasks, chunksize=chunksize))
    else:
        fingerprints = [compute_ecfp_fingerprint(task) for task in tasks]

    valid_mask = [fp is not None for fp in fingerprints]
    clean_dataframe = dataframe.loc[valid_mask].reset_index(drop=True)
    fingerprint_matrix = np.stack([fp for fp in fingerprints if fp is not None]).astype(np.float32)
    return clean_dataframe, fingerprint_matrix


def bipolarize(tensor: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    """{0,1} -> {-1,+1} (o cualquier real: negativo/cero -> -1, positivo -> +1).

    Los +1/-1 van como escalares y no como `ones_like`: la version anterior
    materializaba dos tensores del tamano completo de la entrada (mas la
    mascara y el resultado) solo para escribir constantes, lo que triplicaba
    el pico de memoria justo sobre el tensor mas grande del pipeline.
    `dtype` permite pedir la salida en int8, que es todo lo que hace falta
    para guardar valores en {-1, +1}."""
    target_dtype = tensor.dtype if dtype is None else dtype
    return torch.where(
        tensor > 0,
        torch.tensor(1, dtype=target_dtype),
        torch.tensor(-1, dtype=target_dtype),
    )


def build_projection_matrix(n_bits: int, dim: int, random_state: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(random_state)
    coin_flips = torch.bernoulli(torch.full((n_bits, dim), 0.5), generator=generator)
    return coin_flips * 2 - 1


def encode_fingerprints(
    fingerprint_matrix: np.ndarray,
    projection: torch.Tensor,
    batch_rows: int = HV_BATCH_ROWS,
) -> torch.Tensor:
    """fingerprint (0/1) -> hipervector bipolar {-1,+1} de dimension `dim`.

    Se proyecta por lotes de filas y el resultado se guarda en int8. Antes se
    materializaba la matriz proyectada entera en float32 (para HIV: 41127 x
    10048 = 1,65 GB) y ademas cada slice train/test de las 100 iteraciones se
    llevaba otro tanto. Los hipervectores solo valen -1 o +1, asi que int8
    alcanza y baja ese tensor a 413 MB. La proyeccion intermedia se sigue
    haciendo en float32, de modo que los valores resultantes son identicos a
    los de la version que calculaba todo de una."""
    num_rows = int(fingerprint_matrix.shape[0])
    dim = int(projection.shape[1])
    hvs = torch.empty((num_rows, dim), dtype=torch.int8)
    for start in range(0, num_rows, batch_rows):
        end = min(start + batch_rows, num_rows)
        chunk = torch.from_numpy(fingerprint_matrix[start:end])
        projected = bipolarize(chunk) @ projection
        hvs[start:end] = bipolarize(projected, dtype=torch.int8)
    return hvs


def sum_hvs_in_batches(hvs: torch.Tensor, batch_rows: int = HV_BATCH_ROWS) -> torch.Tensor:
    """Suma hipervectores fila a fila acumulando en float32, por lotes.

    Sumar el tensor entero de una obliga a convertir todo el int8 a float
    (para HIV: 1,32 GB solo del train). El acumulado es exacto igual: los
    valores son enteros en {-1, +1} y la suma nunca supera la cantidad de
    moleculas, muy por debajo de los 2^24 que float32 representa sin perdida,
    asi que el orden de acumulacion no altera el resultado."""
    total = torch.zeros(hvs.shape[1], dtype=torch.float32)
    for start in range(0, int(hvs.shape[0]), batch_rows):
        end = min(start + batch_rows, int(hvs.shape[0]))
        total += hvs[start:end].to(torch.float32).sum(dim=0)
    return total


class HDBindClassifier:
    """Memoria asociativa binaria con reentrenamiento tipo perceptron
    (single-pass bundling + correccion iterativa de errores), clasificando
    por similitud coseno contra el vector de cada clase."""

    def __init__(self, dim: int, num_classes: int = 2) -> None:
        self.dim = dim
        self.num_classes = num_classes
        self.am = torch.zeros(num_classes, dim, dtype=torch.float32)

    def build_am(self, train_hvs: torch.Tensor, train_labels: torch.Tensor) -> None:
        self.am = torch.zeros(self.num_classes, self.dim, dtype=torch.float32)
        for class_label in range(self.num_classes):
            mask = train_labels == class_label
            if mask.any():
                self.am[class_label] = sum_hvs_in_batches(train_hvs[mask])

    def scores(self, hvs: torch.Tensor) -> torch.Tensor:
        """Coseno contra el vector de cada clase, por lotes de filas.

        `hvs` viene en int8 (ver `encode_fingerprints`) y hay que pasarlo a
        float para normalizar; convertir el tensor entero de una vuelve a
        pedir los 1,65 GB que el int8 justamente evita. Cada fila se normaliza
        y multiplica de forma independiente, asi que lotear no cambia el
        resultado."""
        am_norm = torch.nn.functional.normalize(self.am, dim=1)
        num_rows = int(hvs.shape[0])
        out = torch.empty((num_rows, self.num_classes), dtype=torch.float32)
        for start in range(0, num_rows, HV_BATCH_ROWS):
            end = min(start + HV_BATCH_ROWS, num_rows)
            chunk = hvs[start:end].to(torch.float32)
            out[start:end] = torch.nn.functional.normalize(chunk, dim=1) @ am_norm.T
        return out

    def predict(self, hvs: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.scores(hvs), dim=1)

    def retrain_epoch(self, train_hvs: torch.Tensor, train_labels: torch.Tensor) -> int:
        predictions = self.predict(train_hvs)
        mistakes = predictions != train_labels
        mistake_count = int(mistakes.sum().item())
        if mistake_count == 0:
            return 0
        for hv, true_label, predicted_label in zip(
            train_hvs[mistakes], train_labels[mistakes], predictions[mistakes]
        ):
            self.am[int(true_label)] += hv
            self.am[int(predicted_label)] -= hv
        return mistake_count

    def fit(self, train_hvs: torch.Tensor, train_labels: torch.Tensor, num_epochs: int) -> list[int]:
        self.build_am(train_hvs, train_labels)
        learning_curve = []
        for _ in range(num_epochs):
            mistake_count = self.retrain_epoch(train_hvs, train_labels)
            learning_curve.append(mistake_count)
            if mistake_count == 0:
                break
        return learning_curve


def score_for_predictions_csv(scores: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Probabilidad/score a guardar en las predicciones: para binario, el
    score de la clase positiva (comportamiento historico); para multiclase
    , el score de la clase predicha."""
    if scores.shape[1] == 2:
        return scores[:, 1]
    return scores[np.arange(len(y_pred)), y_pred]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    """scores: matriz completa de similitud coseno por clase (n_samples,
    n_classes). Si el target tiene mas de 2 clases , se usa promedio macro y ROC AUC one-vs-rest."""
    is_multiclass = scores.shape[1] > 2
    average = "macro" if is_multiclass else "binary"
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        if is_multiclass:
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores, multi_class="ovr", average="macro"))
        else:
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores[:, 1]))
    except ValueError:
        metrics["roc_auc"] = 0.5
    return metrics


def average_metric(values: list[float]) -> float:
    return float(sum(values) / len(values))


def std_metric(values: list[float]) -> float:
    """Desviacion estandar muestral (ddof=1). Con 1 sola iteracion no hay
    variabilidad que medir, asi que devuelve 0.0 en vez de dividir por 0."""
    if len(values) < 2:
        return 0.0
    mean = average_metric(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return float(variance ** 0.5)


def average_confusion_matrix(matrices: list[list[list[int]]]) -> list[list[float]]:
    return np.asarray(matrices, dtype=float).mean(axis=0).tolist()


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
        "retrain_mistakes_list": [],
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
        "training_seconds": tracker["training_seconds_list"][idx],
        "testing_seconds": tracker["testing_seconds_list"][idx],
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


def run_training_loop(
    hvs: torch.Tensor,
    labels: np.ndarray,
    clean_dataframe: pd.DataFrame,
    id_column: str,
    dim: int,
    num_epochs: int,
    test_size: float,
    random_state: int,
    iterations: int,
    num_classes: int = 2,
    groups: np.ndarray | None = None,
) -> dict[str, object]:
    process = psutil.Process() if psutil is not None else None
    tracker = create_metrics_tracker()
    detailed_rows = []
    best_predictions = None
    best_auroc = float("-inf")
    train_size_used = None
    test_size_used = None

    all_indices = np.arange(hvs.shape[0])

    for iteration in range(iterations):
        iteration_seed = random_state + iteration
        start_time = time.time()
        cpu_time_start = get_process_cpu_time_seconds(process)

        if groups is None:
            train_idx, test_idx = train_test_split(
                all_indices,
                test_size=test_size,
                random_state=iteration_seed,
                stratify=labels,
            )
        else:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=iteration_seed)
            train_idx, test_idx = next(splitter.split(all_indices, labels, groups=groups))

        train_hvs = hvs[train_idx]
        test_hvs = hvs[test_idx]
        train_labels = torch.from_numpy(labels[train_idx])
        y_test = labels[test_idx]

        training_start = time.time()
        classifier = HDBindClassifier(dim=dim, num_classes=num_classes)
        learning_curve = classifier.fit(train_hvs, train_labels, num_epochs=num_epochs)
        training_seconds = time.time() - training_start

        testing_start = time.time()
        scores = classifier.scores(test_hvs)
        y_pred = torch.argmax(scores, dim=1).numpy()
        scores_np = scores.numpy()
        y_score = score_for_predictions_csv(scores_np, y_pred)
        metrics = compute_metrics(y_test, y_pred, scores_np)
        testing_seconds = time.time() - testing_start

        elapsed_seconds = time.time() - start_time
        cpu_time_seconds, memory_percent, memory_mb = measure_resources(process, cpu_time_start)

        tracker["accuracy_list"].append(metrics["accuracy"])
        tracker["auroc_list"].append(metrics["roc_auc"])
        tracker["bacc_list"].append(metrics["balanced_accuracy"])
        tracker["f1_list"].append(metrics["f1"])
        tracker["precision_list"].append(metrics["precision"])
        tracker["recall_list"].append(metrics["recall"])
        tracker["confusion_matrices"].append(metrics["confusion_matrix"])
        tracker["random_states"].append(iteration_seed)
        tracker["cpu_time_seconds_list"].append(cpu_time_seconds)
        tracker["memory_percent_list"].append(memory_percent)
        tracker["memory_mb_list"].append(memory_mb)
        tracker["elapsed_seconds_list"].append(float(elapsed_seconds))
        tracker["training_seconds_list"].append(float(training_seconds))
        tracker["testing_seconds_list"].append(float(testing_seconds))
        tracker["retrain_mistakes_list"].append(learning_curve)

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
                "retrain_epochs_run": len(learning_curve),
                "final_epoch_mistakes": learning_curve[-1] if learning_curve else None,
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
            best_predictions = pd.DataFrame(
                {
                    "id": clean_dataframe.iloc[test_idx][id_column].to_numpy(),
                    "y_true": y_test,
                    "y_pred": y_pred,
                    "score_positive_class": y_score,
                    "iteration": iteration + 1,
                    "random_state": iteration_seed,
                }
            )
            train_size_used = len(train_idx)
            test_size_used = len(test_idx)

        print(
            f"[{iteration + 1}/{iterations}] accuracy={metrics['accuracy']:.4f} "
            f"auroc={metrics['roc_auc']:.4f} balanced_accuracy={metrics['balanced_accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
            f"random_state={iteration_seed} retrain_epochs={len(learning_curve)} "
            f"cpu_time={cpu_time_seconds:.2f}s memory={memory_mb:.1f}MB elapsed={elapsed_seconds:.2f}s"
        )

    average_metrics = {
        "accuracy": average_metric(tracker["accuracy_list"]),
        "roc_auc": average_metric(tracker["auroc_list"]),
        "balanced_accuracy": average_metric(tracker["bacc_list"]),
        "f1": average_metric(tracker["f1_list"]),
        "precision": average_metric(tracker["precision_list"]),
        "recall": average_metric(tracker["recall_list"]),
        "confusion_matrix": average_confusion_matrix(tracker["confusion_matrices"]),
    }

    return {
        "metrics_dict": tracker,
        "detailed_rows": detailed_rows,
        "best_predictions": best_predictions,
        "train_size_used": train_size_used,
        "test_size_used": test_size_used,
        "average_metrics": average_metrics,
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    overall_start = time.time()
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    n_jobs = resolve_n_jobs(args.n_jobs)
    dataset_stem = dataset_path.stem.lower()

    print(f"Usando {n_jobs} nucleos para calcular fingerprints ECFP (radio={args.ecfp_radius}, bits={args.ecfp_bits}).")

    frac_columns = (
        (args.frac_a_column, args.frac_b_column)
        if args.frac_a_column and args.frac_b_column
        else None
    )
    extra_columns = tuple(
        column for column in (args.frac_a_column, args.frac_b_column, args.group_column) if column
    )

    dataframe = load_dataset_table(
        dataset_path=dataset_path,
        smiles_column=args.smiles_column,
        target_column=args.target_column,
        id_column=args.id_column,
        extra_columns=extra_columns,
    )

    # Fingerprints con RDKit via multiprocessing (fork) ANTES de tocar torch:
    # asi ningun worker hereda estado de threads de torch del proceso padre.
    clean_dataframe, fingerprint_matrix = build_fingerprint_table(
        dataframe=dataframe,
        smiles_column=args.smiles_column,
        radius=args.ecfp_radius,
        n_bits=args.ecfp_bits,
        n_jobs=n_jobs,
    )
    labels = clean_dataframe[args.target_column].to_numpy(dtype=np.int64)
    num_classes = int(len(np.unique(labels)))
    groups = clean_dataframe[args.group_column].to_numpy() if args.group_column else None

    print(f"Dataset original: {len(dataframe)} filas")
    print(f"Dataset con fingerprint ECFP valido: {len(clean_dataframe)} filas")
    for class_label in range(num_classes):
        print(f"Clase {class_label}: {int((labels == class_label).sum())}", end="  ")
    print()

    if frac_columns:
        fracs = clean_dataframe[[frac_columns[0], frac_columns[1]]].to_numpy(dtype=np.float32)
        fingerprint_matrix = np.concatenate([fingerprint_matrix, fracs], axis=1)

    # Encoding: se calcula una unica vez para todas las moleculas (no depende
    # de la iteracion), igual que graphHD con sus hipervectores de grafo.
    encoding_start = time.time()
    projection = build_projection_matrix(n_bits=fingerprint_matrix.shape[1], dim=args.dim, random_state=args.random_state)
    hvs = encode_fingerprints(fingerprint_matrix, projection)
    print(f"Hipervectores ({args.dim}d) calculados para {len(clean_dataframe)} moleculas en {time.time() - encoding_start:.2f}s")

    print(f"Iteraciones: {args.iterations}")
    print(f"Epocas de reentrenamiento por iteracion: {args.num_epochs}")
    print(f"Split por iteracion: 80/20 con seed {args.random_state} + iteracion")

    training_result = run_training_loop(
        hvs=hvs,
        labels=labels,
        clean_dataframe=clean_dataframe,
        id_column=args.id_column,
        dim=args.dim,
        num_epochs=args.num_epochs,
        test_size=args.test_size,
        random_state=args.random_state,
        iterations=args.iterations,
        num_classes=num_classes,
        groups=groups,
    )

    tracker = training_result["metrics_dict"]
    best_idx = tracker["auroc_list"].index(max(tracker["auroc_list"]))
    worst_idx = tracker["auroc_list"].index(min(tracker["auroc_list"]))

    summary = {
        "pipeline": "hdbind",
        "dataset_path": str(dataset_path),
        "total_rows_original": int(len(dataframe)),
        "total_rows_valid_rdkit": int(len(clean_dataframe)),
        "train_size": int(training_result["train_size_used"]),
        "test_size": int(training_result["test_size_used"]),
        "smiles_column": args.smiles_column,
        "target_column": args.target_column,
        "random_state_base": args.random_state,
        "test_fraction": args.test_size,
        "iterations": args.iterations,
        "dim": args.dim,
        "ecfp_radius": args.ecfp_radius,
        "ecfp_bits": args.ecfp_bits,
        "num_epochs": args.num_epochs,
        "metrics_by_iteration": tracker,
        "average_metrics": training_result["average_metrics"],
        "average_resources": average_resources_from_tracker(tracker),
        "best_run": run_snapshot_from_tracker(tracker, best_idx),
        "worst_run": run_snapshot_from_tracker(tracker, worst_idx),
        "timing_summary": {
            "total_wall_clock_seconds": float(time.time() - overall_start),
            "total_training_seconds": float(sum(tracker["training_seconds_list"])),
            "total_testing_seconds": float(sum(tracker["testing_seconds_list"])),
        },
    }

    metrics_json = output_dir / f"{dataset_stem}_hdbind_metrics.json"
    detailed_csv = output_dir / f"{dataset_stem}_hdbind_detailed_results.csv"
    predictions_csv = output_dir / f"{dataset_stem}_hdbind_best_iteration_predictions.csv"

    save_json(metrics_json, summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(training_result["detailed_rows"]).to_csv(detailed_csv, index=False)
    training_result["best_predictions"].to_csv(predictions_csv, index=False)

    print()
    print(f"Average Stats for {args.iterations} iterations")
    average_metrics = training_result["average_metrics"]
    print("Accuracy: ", average_metrics["accuracy"])
    print("Auroc: ", average_metrics["roc_auc"])
    print("Bacc: ", average_metrics["balanced_accuracy"])
    print("F1: ", average_metrics["f1"])
    print("Precision: ", average_metrics["precision"])
    print("Recall: ", average_metrics["recall"])
    print()
    print(f"Standard Deviation for {args.iterations} iterations")
    print("Accuracy: ", std_metric(tracker["accuracy_list"]))
    print("Auroc: ", std_metric(tracker["auroc_list"]))
    print("Bacc: ", std_metric(tracker["bacc_list"]))
    print("F1: ", std_metric(tracker["f1_list"]))
    print("Precision: ", std_metric(tracker["precision_list"]))
    print("Recall: ", std_metric(tracker["recall_list"]))
    print()
    print("Tiempos totales:")
    print(f"Tiempo total: {summary['timing_summary']['total_wall_clock_seconds']:.2f}s")
    print(f"Tiempo total de entrenamiento: {summary['timing_summary']['total_training_seconds']:.2f}s")
    print(f"Tiempo total de testeo: {summary['timing_summary']['total_testing_seconds']:.2f}s")
    print()
    print(f"Metricas guardadas en: {metrics_json}")
    print(f"Resultados detallados guardados en: {detailed_csv}")
    print(f"Predicciones mejor iteracion guardadas en: {predictions_csv}")


if __name__ == "__main__":
    main()
