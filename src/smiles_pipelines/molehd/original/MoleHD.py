from SmilesPE.tokenizer import *
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from scipy.spatial.distance import cdist
import sklearn.metrics
import imblearn
from tqdm import tqdm, trange
from utils import *
import pickle
import json
import os, argparse
import random
import psutil
import time
import concurrent.futures
from rdkit import Chem

DEFAULT_RANDOM_STATE = 800


def get_process_cpu_time_seconds(process):
    cpu_times = process.cpu_times()
    return float(cpu_times.user + cpu_times.system)


def _default_worker_count(dataset_size, dim):
    """Limita los workers segun la RAM disponible en vez de un numero fijo.
    Cada worker mantiene en memoria data_HV (dataset_size x dim, float64)
    mas el set de entrenamiento oversampleado por RandomOverSampler, que en
    datasets muy desbalanceados (ej. HIV: 1443 positivos contra 39684
    negativos) casi duplica el train set. Con datasets grandes (HIV tiene
    5x mas filas que otros datasets chicos del catalogo) un numero fijo de
    workers que andaba bien en un dataset chico revienta la RAM del contenedor en uno grande
    (BrokenProcessPool)."""
    cpu_workers = os.cpu_count() or 1
    try:
        available = psutil.virtual_memory().available
    except Exception:
        return min(cpu_workers, 6)

    bytes_per_hv_matrix = dataset_size * dim * 8
    per_worker_estimate = int(bytes_per_hv_matrix * 4 + 512 * 1024 * 1024)
    usable_memory = available * 0.5
    mem_workers = max(1, int(usable_memory // per_worker_estimate))
    return max(1, min(cpu_workers, mem_workers, 8))


def _tokenize(X, encoding_scheme, num_tokens):
    if encoding_scheme.lower() == "smiles_pretrained":
        return data_tokenize_smiles_pretrained(X, num_tokens=num_tokens)
    elif encoding_scheme.lower() == "atomwise":
        return data_tokenize_atomwise(X, num_tokens=num_tokens)
    elif encoding_scheme.lower() == "characterwise":
        return data_tokenize_characterwise(X, num_tokens=num_tokens)
    raise ValueError(f"MoleHD currently do not support {encoding_scheme} encoding scheme.")


def _run_single_iteration(payload):
    (iteration, X, Y, num_tokens, dim, max_pos, gramsize, epochs, threshold,
     encoding_scheme, split_type, test_size, groups, num_classes) = payload

    iteration_start = time.time()
    random_state = DEFAULT_RANDOM_STATE + iteration

    data_tokenized = _tokenize(X, encoding_scheme, num_tokens)
    data_HV = create_data_HV(data_tokenized, gramsize=gramsize, num_tokens=num_tokens, dim=dim, max_pos=max_pos, random_state=random_state)

    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size / 100, random_state=random_state)
        all_indices = np.arange(len(X))
        train_idx, test_idx = next(splitter.split(all_indices, Y, groups=groups))
        X_tr = [data_HV[i] for i in train_idx]
        X_te = [data_HV[i] for i in test_idx]
        Y_tr = [Y[i] for i in train_idx]
        Y_te = [Y[i] for i in test_idx]
    elif split_type.lower() == "scaffold":
        X_tr, X_te, Y_tr, Y_te = train_test_split_scaffold(X, Y, data_HV, test_size=test_size / 100, random_state=random_state)
    elif split_type.lower() == "random":
        X_tr, X_te, Y_tr, Y_te = train_test_split(data_HV, Y, test_size=test_size / 100, random_state=random_state)
    elif split_type.lower() == "random_stratified":
        X_tr, X_te, Y_tr, Y_te = train_test_split(data_HV, Y, test_size=test_size / 100, random_state=random_state, stratify=Y)
    else:
        raise ValueError(f"MoleHD currently do not support {split_type} split type.")

    process = psutil.Process(os.getpid())
    iteration_cpu_start = get_process_cpu_time_seconds(process)

    training_start = time.time()
    # 'not majority' oversamplea todas las clases salvo la mayoritaria; para
    # 2 clases es identico a 'minority' (no cambia el comportamiento binario
    # existente), y generaliza correctamente a N clases.
    # random_state explicito: sin el, RandomOverSampler cae al RNG global de
    # numpy para elegir que muestras duplicar, que en un worker de
    # ProcessPoolExecutor no esta sembrado -- misma clase de bug que tenia
    # create_data_HV. Con esto la iteracion queda determinada por completo por
    # su semilla, que es lo que hace falta para que dos corridas coincidan.
    oversample = imblearn.over_sampling.RandomOverSampler(
        sampling_strategy='not majority', random_state=random_state
    )
    X_tr, Y_tr = oversample.fit_resample(X_tr, Y_tr)
    X_tr = np.array(X_tr)

    assoc_mem = np.zeros((num_classes, dim))
    for i in range(len(Y_tr)):
        assoc_mem[Y_tr[i]] += X_tr[i]
    assoc_mem = retrain(assoc_mem, X_tr, Y_tr, epochs=epochs, dim=dim, threshold=threshold)
    training_seconds = time.time() - training_start

    testing_start = time.time()
    Y_pred, Y_scores = inference(assoc_mem, X_te, Y_te, dim=dim)
    testing_seconds = time.time() - testing_start

    is_multiclass = num_classes > 2
    average = "macro" if is_multiclass else "binary"
    try:
        if is_multiclass:
            auroc = sklearn.metrics.roc_auc_score(Y_te, Y_scores, multi_class="ovr", average="macro")
        else:
            auroc = sklearn.metrics.roc_auc_score(Y_te, Y_scores[:, 1])
    except ValueError:
        auroc = 0.5

    cpu_time_seconds = max(0.0, get_process_cpu_time_seconds(process) - iteration_cpu_start)
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    memory_mb = memory_info.rss / 1024 / 1024

    return iteration, {
        "accuracy": sklearn.metrics.accuracy_score(Y_te, Y_pred),
        "auroc": auroc,
        "bacc": sklearn.metrics.balanced_accuracy_score(Y_te, Y_pred),
        "f1": sklearn.metrics.f1_score(Y_te, Y_pred, average=average, zero_division=0),
        "precision": sklearn.metrics.precision_score(Y_te, Y_pred, average=average, zero_division=0),
        "recall": sklearn.metrics.recall_score(Y_te, Y_pred, average=average, zero_division=0),
        "confusion_matrix": sklearn.metrics.confusion_matrix(Y_te, Y_pred),
        "random_state": random_state,
        "elapsed_seconds": time.time() - iteration_start,
        "training_seconds": training_seconds,
        "testing_seconds": testing_seconds,
        "cpu_time_seconds": cpu_time_seconds,
        "memory_percent": memory_percent,
        "memory_mb": memory_mb,
        "assoc_mem": assoc_mem,
    }


def load_dataset(dataset_file, mols, target, extra_columns=()):
    def canon_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol is not None else None

    dataset = pd.read_csv(dataset_file, sep=',', header=0)

    # Some processed Mole-BERT smiles.csv files are headerless, so pandas
    # treats the first SMILES as the column name. Recover that shape here.
    if target not in dataset.columns and mols not in dataset.columns and dataset.shape[1] == 1:
        dataset = pd.read_csv(dataset_file, sep=',', header=None, names=[mols])

    if mols not in dataset.columns:
        if dataset.shape[1] == 1:
            dataset.columns = [mols]
        else:
            raise ValueError(f"Molecule column '{mols}' was not found in {dataset_file}. Available columns: {list(dataset.columns)}")

    if target not in dataset.columns:
        raw_bbbp = os.path.join(os.path.dirname(os.path.dirname(dataset_file)), 'raw', 'BBBP.csv')
        if os.path.isfile(raw_bbbp):
            raw_df = pd.read_csv(raw_bbbp, sep=',', header=0)
            if 'smiles' in raw_df.columns and target in raw_df.columns:
                raw_df = raw_df[['smiles', target]].copy()
                raw_df['smiles'] = raw_df['smiles'].map(canon_smiles)
                raw_df = raw_df.dropna(subset=['smiles'])
                raw_df = raw_df.groupby('smiles', as_index=False)[target].agg(lambda values: int(pd.Series(values).mode().iloc[0]))
                if mols == 'smiles':
                    dataset = dataset.merge(raw_df, on='smiles', how='left')
                else:
                    dataset = dataset.merge(
                        raw_df,
                        left_on=mols,
                        right_on='smiles',
                        how='left'
                    ).drop(columns=['smiles'])

        if target not in dataset.columns:
            raise ValueError(
                f"Target column '{target}' was not found in {dataset_file}. "
                f"If you are using a SMILES-only file, provide or reconstruct labels first."
            )

    missing_extra = [column for column in extra_columns if column not in dataset.columns]
    if missing_extra:
        raise ValueError(f"No encontre estas columnas en {dataset_file}: {missing_extra}")

    dataset = dataset.dropna(subset=[mols, target]).copy()

    unique_targets = set(pd.Series(dataset[target]).dropna().unique().tolist())
    if target == 'Tg (K) exp' and not unique_targets.issubset({0, 1}):
        dataset[target] = dataset[target].apply(lambda x: 0 if x < 350 else 1)
    else:
        # Se acepta cualquier target categorico (binario o multiclase con
        # 3 o mas clases {0,1,2,...}), no solo binario -- la
        # memoria asociativa ahora se dimensiona segun la cantidad real de
        # clases (ver num_classes en __main__). Se rechaza si parece
        # continuo (targets de regresion), no categorico.
        non_integer_values = [value for value in unique_targets if not float(value).is_integer()]
        if non_integer_values:
            raise ValueError(
                f"Target '{target}' tiene valores no enteros ({non_integer_values[:3]}...), parece continuo. "
                "MoleHD.py arma una memoria asociativa de clasificacion: usa un target categorico o discretizalo antes."
            )

    dataset[target] = dataset[target].astype(int)
    return dataset


def average_confusion_matrix(confusion_matrices):
    matrix = np.asarray(confusion_matrices, dtype=float)
    return matrix.mean(axis=0).tolist()


def std_metric(values):
    """Desviacion estandar muestral (ddof=1). Con 1 sola iteracion no hay
    variabilidad que medir, asi que devuelve 0.0 en vez de dividir por 0."""
    values = list(values)
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return float(variance ** 0.5)


def build_canonical_task_summary(tracker, train_size, test_size):
    """Arma average_metrics/best_run/worst_run/average_resources con el
    mismo esquema de nombres (accuracy/roc_auc/balanced_accuracy/f1/
    precision/recall/confusion_matrix, cpu_time_seconds/memory_percent/
    memory_mb/elapsed_seconds) que usan random_forest_rdkit, mole_bert_hdc,
    etc., para que results/all_runs.csv tenga las mismas columnas sin
    importar el pipeline que corrio."""
    best_idx = int(np.argmax(tracker["auroc_list"]))
    worst_idx = int(np.argmin(tracker["auroc_list"]))

    def as_list(value):
        return value.tolist() if hasattr(value, "tolist") else value

    def run_snapshot(idx):
        return {
            "accuracy": tracker["accuracy_list"][idx],
            "roc_auc": tracker["auroc_list"][idx],
            "balanced_accuracy": tracker["bacc_list"][idx],
            "f1": tracker["f1_list"][idx],
            "precision": tracker["precision_list"][idx],
            "recall": tracker["recall_list"][idx],
            "confusion_matrix": as_list(tracker["confusion_matrices"][idx]),
            "random_state": tracker["random_states"][idx],
            "cpu_time_seconds": tracker["cpu_time_seconds_list"][idx],
            "memory_percent": tracker["memory_percent_list"][idx],
            "memory_mb": tracker["memory_mb_list"][idx],
            "elapsed_seconds": tracker["elapsed_seconds_list"][idx],
        }

    return {
        "train_size": train_size,
        "test_size": test_size,
        "average_metrics": {
            "accuracy": sum(tracker["accuracy_list"]) / len(tracker["accuracy_list"]),
            "roc_auc": sum(tracker["auroc_list"]) / len(tracker["auroc_list"]),
            "balanced_accuracy": sum(tracker["bacc_list"]) / len(tracker["bacc_list"]),
            "f1": sum(tracker["f1_list"]) / len(tracker["f1_list"]),
            "precision": sum(tracker["precision_list"]) / len(tracker["precision_list"]),
            "recall": sum(tracker["recall_list"]) / len(tracker["recall_list"]),
            "confusion_matrix": average_confusion_matrix(tracker["confusion_matrices"]),
        },
        "best_run": run_snapshot(best_idx),
        "worst_run": run_snapshot(worst_idx),
        "average_resources": {
            "cpu_time_seconds": sum(tracker["cpu_time_seconds_list"]) / len(tracker["cpu_time_seconds_list"]),
            "memory_percent": sum(tracker["memory_percent_list"]) / len(tracker["memory_percent_list"]),
            "memory_mb": sum(tracker["memory_mb_list"]) / len(tracker["memory_mb_list"]),
            "peak_memory_mb": max(tracker["memory_mb_list"]),
            "elapsed_seconds": sum(tracker["elapsed_seconds_list"]) / len(tracker["elapsed_seconds_list"]),
        },
    }


if __name__ == '__main__':
    overall_start = time.time()

    # initializing all the arguments
    parser = argparse.ArgumentParser(description='MoleHD Framework')
    parser.add_argument('--dataset_file', default='./data/bicerano_bigsmiles.csv', type=str, help="File location. Example, './data/bicerano_bigsmiles.csv' ")
    parser.add_argument('--target', default='Tg (K) exp', type=str, help="Name of target column in file.")
    parser.add_argument('--mols', default='SMILES', type=str, help="Name of column that contains molecules. Use 'SMILES' or 'BigSMILES'.")
    parser.add_argument('--num_tokens', default=500, type=int, help="Number of tokens to be used for data tokenization. Default 1500")
    parser.add_argument('--dim', default=10000, type=int, help="Dimension of hypervector. Default 10000")
    parser.add_argument('--max_pos', default=256, type=int, help="Threshold of position hypervector. Default 256")
    parser.add_argument('--gramsize', default=3, type=int, help="N-gram tokenization size. Default 1")
    parser.add_argument('--retraining_epochs', default=20, type=int, help="Number of iterations to train the model for. Default 150")
    parser.add_argument('--iterations', default=100, type=int, help="Number of iterations to run the entire experiment for. Default 100")
    parser.add_argument('--test_size', default=20, type=int, help="Split percentage for testing set. Defualt 20.")
    parser.add_argument('--threshold', default=256, type=int, help="Threshold to scope the associate memory. Defualt 1024.")
    parser.add_argument('--encoding_scheme', default="characterwise", type=str, help="Encoding scheme for HDC. Supported types [smiles_pretrained, characterwise]")
    parser.add_argument('--split_type', default="random", type=str, help="Data split method. Supported types [scaffold, random, random_stratified]")   
    parser.add_argument('--version', default="v1", type=str, help="Version to be appended to file name while saving model and output.")
    parser.add_argument('--workers', default=None, type=int, help="Cantidad de procesos en paralelo para las iteraciones. Default: todos los cores disponibles.")
    parser.add_argument(
        '--frac_a_column', default=None, type=str,
        help=(
            "Columna opcional con una fraccion de composicion. "
            "Si se pasa junto con --frac_b_column, se cuantiza a decimos y se agrega "
            "como token de sufijo al string antes de tokenizar (ej. '...|FRAC_5_5')."
        ),
    )
    parser.add_argument('--frac_b_column', default=None, type=str, help="Ver --frac_a_column.")
    parser.add_argument(
        '--group_column', default=None, type=str,
        help=(
            "Columna opcional para agrupar el split train/test: "
            "garantiza que un mismo grupo no quede repartido entre train y test."
        ),
    )

    args = parser.parse_args()

    dataset_file = args.dataset_file
    target = args.target
    mols = args.mols
    num_tokens = args.num_tokens
    dim = args.dim
    max_pos = args.max_pos
    gramsize = args.gramsize
    epochs = args.retraining_epochs
    iterations = args.iterations
    test_size = args.test_size
    threshold = args.threshold
    encoding_scheme = args.encoding_scheme
    split_type = args.split_type
    version = args.version

    frac_a_column = args.frac_a_column
    frac_b_column = args.frac_b_column
    group_column = args.group_column
    extra_columns = tuple(column for column in (frac_a_column, frac_b_column, group_column) if column)

    dataset = load_dataset(dataset_file, mols, target, extra_columns=extra_columns)

    X_raw = list(dataset[mols].values)
    Y_raw = list(dataset[target].values)

    # clean_dataset valida SMILES via RDKit; se corre sobre el SMILES crudo
    # (el token de fraccion no es sintaxis SMILES valida) y se recuperan los
    # indices originales para poder alinear frac_a/frac_b/group despues.
    original_indices = list(range(len(X_raw)))
    X_clean, kept_indices, X_bad, _ = clean_dataset(X_raw, original_indices)
    Y = [Y_raw[idx] for idx in kept_indices]

    if frac_a_column and frac_b_column:
        frac_a_values = dataset[frac_a_column].to_numpy()
        frac_b_values = dataset[frac_b_column].to_numpy()
        X = [
            f"{smiles}|FRAC_{int(round(float(frac_a_values[idx]) * 10))}_{int(round(float(frac_b_values[idx]) * 10))}"
            for smiles, idx in zip(X_clean, kept_indices)
        ]
    else:
        X = X_clean

    groups = dataset[group_column].to_numpy()[kept_indices] if group_column else None
    num_classes = len(set(Y))

    print(len(X), len(Y))
    class_counts = Counter(Y)
    print("  ".join(f"Clase {label}: {class_counts[label]}" for label in sorted(class_counts)))
    if X_bad:
        print(f"Moléculas descartadas por validación: {len(X_bad)}")
    print(f"Iteraciones: {iterations}")
    print(f"Split por iteración: {int((1 - test_size/100) * 100)}/{test_size} con seed 800 + iteración")

    accuracy_list = []
    auroc_list = []
    bacc_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    confusion_matrices = []

    metrics_dict = dict()
    metrics_dict["accuracy_list"] = list()
    metrics_dict["auroc_list"] = list()
    metrics_dict["bacc_list"] = list()
    metrics_dict["f1_list"] = list()
    metrics_dict["precision_list"] = list()
    metrics_dict["recall_list"] = list()
    metrics_dict["confusion_matrices"] = list()
    metrics_dict["random_states"] = list()
    metrics_dict["cpu_time_seconds_list"] = list()
    metrics_dict["memory_percent_list"] = list()
    metrics_dict["memory_mb_list"] = list()
    metrics_dict["elapsed_seconds_list"] = list()
    metrics_dict["training_seconds_list"] = list()
    metrics_dict["testing_seconds_list"] = list()

    max_assoc_mem = []
    max_auroc = 0

    max_workers = args.workers or _default_worker_count(len(X), dim)
    print(f"Corriendo {iterations} iteraciones en paralelo con {max_workers} workers")

    worker_payloads = [
        (iteration, X, Y, num_tokens, dim, max_pos, gramsize, epochs, threshold,
         encoding_scheme, split_type, test_size, groups, num_classes)
        for iteration in range(iterations)
    ]

    results_by_iteration = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_single_iteration, payload) for payload in worker_payloads]
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            iteration, res = future.result()
            results_by_iteration[iteration] = res
            print(f"Iteracion {done}/{len(futures)} completada.")

    for iteration in range(iterations):
        res = results_by_iteration[iteration]

        if res["auroc"] > max_auroc:
            max_assoc_mem = res["assoc_mem"]
            max_auroc = res["auroc"]

        metrics_dict["accuracy_list"].append(res["accuracy"])
        metrics_dict["auroc_list"].append(res["auroc"])
        metrics_dict["bacc_list"].append(res["bacc"])
        metrics_dict["f1_list"].append(res["f1"])
        metrics_dict["precision_list"].append(res["precision"])
        metrics_dict["recall_list"].append(res["recall"])
        metrics_dict["confusion_matrices"].append(res["confusion_matrix"])
        metrics_dict["random_states"].append(res["random_state"])
        metrics_dict["elapsed_seconds_list"].append(res["elapsed_seconds"])
        metrics_dict["training_seconds_list"].append(res["training_seconds"])
        metrics_dict["testing_seconds_list"].append(res["testing_seconds"])

        metrics_dict["cpu_time_seconds_list"].append(res["cpu_time_seconds"])
        metrics_dict["memory_percent_list"].append(res["memory_percent"])
        metrics_dict["memory_mb_list"].append(res["memory_mb"])
        assoc_mem = res["assoc_mem"]

        print(
            f"[{iteration + 1}/{iterations}] accuracy={res['accuracy']:.4f} "
            f"auroc={res['auroc']:.4f} bacc={res['bacc']:.4f} f1={res['f1']:.4f} "
            f"precision={res['precision']:.4f} recall={res['recall']:.4f} "
            f"random_state={res['random_state']} cpu_time={res['cpu_time_seconds']:.2f}s "
            f"memory={res['memory_mb']:.1f}MB elapsed={res['elapsed_seconds']:.2f}s"
        )

    print()
    print("Stats corresponding to Maximum AUROC are: ")

    max_auroc = max(metrics_dict["auroc_list"])
    max_auroc_idx = metrics_dict["auroc_list"].index(max_auroc)
    print("Accuracy: ", metrics_dict["accuracy_list"][max_auroc_idx])
    print("Auroc: ", metrics_dict["auroc_list"][max_auroc_idx])
    print("Bacc: ", metrics_dict["bacc_list"][max_auroc_idx])
    print("F1: ", metrics_dict["f1_list"][max_auroc_idx])
    print("Precision: ", metrics_dict["precision_list"][max_auroc_idx])
    print("Recall: ", metrics_dict["recall_list"][max_auroc_idx])
    print("Confusion Matrix: ", metrics_dict["confusion_matrices"][max_auroc_idx])
    print("Random State: ", metrics_dict["random_states"][max_auroc_idx])
    print("CPU Total Time: {:.2f}s".format(metrics_dict["cpu_time_seconds_list"][max_auroc_idx]))
    print("Memory Usage: {:.2f}% ({:.2f} MB)".format(metrics_dict["memory_percent_list"][max_auroc_idx], metrics_dict["memory_mb_list"][max_auroc_idx]))

    print()

    print("Stats corresponding to Minimum AUROC are: ")

    min_auroc = min(metrics_dict["auroc_list"])
    min_auroc_idx = metrics_dict["auroc_list"].index(min_auroc)
    print("Accuracy: ", metrics_dict["accuracy_list"][min_auroc_idx])
    print("Auroc: ", metrics_dict["auroc_list"][min_auroc_idx])
    print("Bacc: ", metrics_dict["bacc_list"][min_auroc_idx])
    print("F1: ", metrics_dict["f1_list"][min_auroc_idx])
    print("Precision: ", metrics_dict["precision_list"][min_auroc_idx])
    print("Recall: ", metrics_dict["recall_list"][min_auroc_idx])
    print("Confusion Matrix: ", metrics_dict["confusion_matrices"][min_auroc_idx])
    print("Random State: ", metrics_dict["random_states"][min_auroc_idx])
    print("CPU Total Time: {:.2f}s".format(metrics_dict["cpu_time_seconds_list"][min_auroc_idx]))
    print("Memory Usage: {:.2f}% ({:.2f} MB)".format(metrics_dict["memory_percent_list"][min_auroc_idx], metrics_dict["memory_mb_list"][min_auroc_idx]))

    print()

    print(f"Average Stats for {iterations} iterations")
    print("Accuracy: ", sum(metrics_dict["accuracy_list"])/iterations)
    print("Auroc: ", sum(metrics_dict["auroc_list"])/iterations)
    print("Bacc: ", sum(metrics_dict["bacc_list"])/iterations)
    print("F1: ", sum(metrics_dict["f1_list"])/iterations)
    print("Precision: ", sum(metrics_dict["precision_list"])/iterations)
    print("Recall: ", sum(metrics_dict["recall_list"])/iterations)
    print()
    print(f"Standard Deviation for {iterations} iterations")
    print("Accuracy: ", std_metric(metrics_dict["accuracy_list"]))
    print("Auroc: ", std_metric(metrics_dict["auroc_list"]))
    print("Bacc: ", std_metric(metrics_dict["bacc_list"]))
    print("F1: ", std_metric(metrics_dict["f1_list"]))
    print("Precision: ", std_metric(metrics_dict["precision_list"]))
    print("Recall: ", std_metric(metrics_dict["recall_list"]))
    print("Average CPU Total Time: {:.2f}s".format(sum(metrics_dict["cpu_time_seconds_list"])/iterations))
    print("Average Memory Usage: {:.2f}% ({:.2f} MB)".format(sum(metrics_dict["memory_percent_list"])/iterations, sum(metrics_dict["memory_mb_list"])/iterations))
    print("Peak Memory Usage: {:.2f} MB".format(max(metrics_dict["memory_mb_list"])))
    print()
    print("Tiempos totales:")
    print("Tiempo total: {:.2f}s".format(time.time() - overall_start))
    print("Tiempo total de entrenamiento: {:.2f}s".format(sum(metrics_dict["training_seconds_list"])))
    print("Tiempo total de testeo: {:.2f}s".format(sum(metrics_dict["testing_seconds_list"])))

    dataset_file_suffix = dataset_file.split("/")[-1].split(".")[0]
    file_suffix = f"{dataset_file_suffix}_data_{target}_tar_{dim}_dim_{gramsize}_gm_{encoding_scheme}_{split_type}_{version}.p"

    os.makedirs('./outputs', exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    print()
    print("Saving performance metrics dictionary and best performing model...")

    with open(f'./outputs/metrics_dict_{file_suffix}', 'wb') as f:
        pickle.dump(metrics_dict, f)

    with open(f'./models/model_{file_suffix}', 'wb') as f:
        pickle.dump(max_assoc_mem if len(max_assoc_mem) else assoc_mem, f)

    canonical_train_size = int(round(len(X) * (1.0 - test_size / 100.0)))
    canonical_summary = build_canonical_task_summary(
        metrics_dict, train_size=canonical_train_size, test_size=len(X) - canonical_train_size
    )
    canonical_payload = {
        "pipeline": "molehd",
        "dataset_path": dataset_file,
        "total_rows_original": len(dataset),
        "total_rows_valid_rdkit": len(X),
        "train_size": canonical_summary["train_size"],
        "test_size": canonical_summary["test_size"],
        "smiles_column": mols,
        "target_column": target,
        "random_state_base": DEFAULT_RANDOM_STATE,
        "test_fraction": test_size / 100.0,
        "iterations": iterations,
        "dim": dim,
        "average_metrics": canonical_summary["average_metrics"],
        "best_run": canonical_summary["best_run"],
        "worst_run": canonical_summary["worst_run"],
        "average_resources": canonical_summary["average_resources"],
        "timing_summary": {
            "total_wall_clock_seconds": float(time.time() - overall_start),
            "total_training_seconds": float(sum(metrics_dict["training_seconds_list"])),
            "total_testing_seconds": float(sum(metrics_dict["testing_seconds_list"])),
        },
    }
    json_file_suffix = file_suffix[:-2] if file_suffix.endswith(".p") else file_suffix
    with open(f'./outputs/metrics_{json_file_suffix}.json', 'w', encoding='utf-8') as f:
        json.dump(canonical_payload, f, indent=2)

    # Export detailed results to CSV (one row per iteration)
    import csv
    csv_filename = f'./outputs/detailed_results_{file_suffix}.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Iteration', 'Accuracy', 'AUROC', 'Balanced_Accuracy', 'F1_Score',
                      'Precision', 'Recall', 'CPU_Total_Time_Seconds', 'Memory_Usage_%', 'Memory_MB',
                      'Random_State', 'Elapsed_Seconds', 'Training_Seconds', 'Testing_Seconds']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(iterations):
            writer.writerow({
                'Iteration': i + 1,
                'Accuracy': metrics_dict["accuracy_list"][i],
                'AUROC': metrics_dict["auroc_list"][i],
                'Balanced_Accuracy': metrics_dict["bacc_list"][i],
                'F1_Score': metrics_dict["f1_list"][i],
                'Precision': metrics_dict["precision_list"][i],
                'Recall': metrics_dict["recall_list"][i],
                'CPU_Total_Time_Seconds': metrics_dict["cpu_time_seconds_list"][i],
                'Memory_Usage_%': metrics_dict["memory_percent_list"][i],
                'Memory_MB': metrics_dict["memory_mb_list"][i],
                'Random_State': metrics_dict["random_states"][i],
                'Elapsed_Seconds': metrics_dict["elapsed_seconds_list"][i],
                'Training_Seconds': metrics_dict["training_seconds_list"][i],
                'Testing_Seconds': metrics_dict["testing_seconds_list"][i],
            })
    
    print(f"Detailed results saved to {csv_filename}")
    print(f"Metricas (json) guardadas en: ./outputs/metrics_{json_file_suffix}.json")
    print("Saving completed.")
