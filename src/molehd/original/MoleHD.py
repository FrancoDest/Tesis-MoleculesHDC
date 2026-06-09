from SmilesPE.tokenizer import *
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import sklearn.metrics 
import imblearn
from tqdm import tqdm, trange
from utils import *
import pickle
import os, argparse
import random
import psutil
import time
from rdkit import Chem

DEFAULT_RANDOM_STATE = 800


def get_process_cpu_time_seconds(process):
    cpu_times = process.cpu_times()
    return float(cpu_times.user + cpu_times.system)


def load_dataset(dataset_file, mols, target):
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

    dataset = dataset.dropna(subset=[mols, target]).copy()

    unique_targets = set(pd.Series(dataset[target]).dropna().unique().tolist())
    if not unique_targets.issubset({0, 1}):
        if target == 'Tg (K) exp':
            dataset[target] = dataset[target].apply(lambda x: 0 if x < 350 else 1)
        else:
            raise ValueError(
                f"Target '{target}' is not binary and MoleHD.py currently builds a 2-class associative memory. "
                "Use a binary target or adapt the classifier."
            )

    dataset[target] = dataset[target].astype(int)
    return dataset


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

    dataset = load_dataset(dataset_file, mols, target)

    X = list(dataset[mols].values)
    Y = list(dataset[target].values)

    X_clean, Y_clean, X_bad, Y_bad = clean_dataset(X, Y)
    X = X_clean
    Y = Y_clean

    print(len(X), len(Y))
    print(f"Clase 0: {Y.count(0)}, Clase 1: {Y.count(1)}")
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
    process = psutil.Process(os.getpid())

    for iteration in tqdm(range(iterations)):
        
        iteration_start = time.time()
        iteration_cpu_start = get_process_cpu_time_seconds(process)

        random_state = DEFAULT_RANDOM_STATE + iteration
        
        # encoding the molecules into numerical tokens
        if encoding_scheme.lower() == "smiles_pretrained":
            data_tokenized = data_tokenize_smiles_pretrained(X, num_tokens=num_tokens)
        elif encoding_scheme.lower() == "atomwise":
            data_tokenized = data_tokenize_atomwise(X, num_tokens=num_tokens)
        elif encoding_scheme.lower() == "characterwise":
            data_tokenized = data_tokenize_characterwise(X, num_tokens=num_tokens)
        else:
            print(f"MoleHD currently do not support {encoding_scheme} encoding scheme. Please try of of the 3 encoding schemes [smiles_pretrained, atomwise, characterwise]")
        
        # converting numerical tokens representing molecules into a hypervectors
        data_HV = create_data_HV(data_tokenized, gramsize=gramsize, num_tokens=num_tokens, dim=dim, max_pos=max_pos, random_state=random_state)
        
        # splitting the dataset into training and testing based on split type
        if split_type.lower() == "scaffold":
            X_tr, X_te, Y_tr, Y_te = train_test_split_scaffold(X, Y, data_HV, test_size=test_size/100, random_state=random_state)
        elif split_type.lower() == "random":
            X_tr, X_te, Y_tr, Y_te = train_test_split(data_HV, Y, test_size=test_size/100, random_state=random_state)
        elif split_type.lower() == "random_stratified":
            X_tr, X_te, Y_tr, Y_te = train_test_split(data_HV, Y, test_size=test_size/100, random_state=random_state, stratify=Y)
        else:
            print(f"MoleHD currently do not support {split_type} split type. Please try one of the 3 encoding schemes [scaffold, random, random_stratified]")
            
        # oversample to handle data imbalance
        training_start = time.time()
        oversample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='minority') #Oversampling by duplication

        X_tr, Y_tr = oversample.fit_resample(X_tr, Y_tr)
        X_tr = np.array(X_tr)
        
        # Training associative memory
        assoc_mem = np.zeros((2, dim))
        for i in range(len(Y_tr)):
            assoc_mem[Y_tr[i]] += X_tr[i]
            
        # retraining the associative memory to fix misclassifications
        assoc_mem = retrain(assoc_mem, X_tr, Y_tr, epochs=epochs, dim=dim, threshold=threshold)

        training_seconds = time.time() - training_start
        testing_start = time.time()
        Y_pred, Y_score = inference(assoc_mem, X_te, Y_te, dim=dim)
        
        # Metrics
        auroc = sklearn.metrics.roc_auc_score(Y_te, Y_score)
        if auroc > max_auroc:
            max_assoc_mem = assoc_mem
            max_auroc = auroc
        
        metrics_dict["accuracy_list"].append(sklearn.metrics.accuracy_score(Y_te, Y_pred))
        metrics_dict["auroc_list"].append(sklearn.metrics.roc_auc_score(Y_te, Y_score))
        metrics_dict["bacc_list"].append(sklearn.metrics.balanced_accuracy_score(Y_te, Y_pred))
        metrics_dict["f1_list"].append(sklearn.metrics.f1_score(Y_te, Y_pred))
        metrics_dict["precision_list"].append(sklearn.metrics.precision_score(Y_te, Y_pred))
        metrics_dict["recall_list"].append(sklearn.metrics.recall_score(Y_te, Y_pred))
        metrics_dict["confusion_matrices"].append(sklearn.metrics.confusion_matrix(Y_te, Y_pred))
        metrics_dict["random_states"].append(random_state)
        metrics_dict["elapsed_seconds_list"].append(time.time() - iteration_start)
        metrics_dict["training_seconds_list"].append(training_seconds)
        metrics_dict["testing_seconds_list"].append(time.time() - testing_start)
        
        # Resource consumption tracking
        cpu_time_seconds = max(0.0, get_process_cpu_time_seconds(process) - iteration_cpu_start)
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        memory_mb = memory_info.rss / 1024 / 1024
        
        metrics_dict["cpu_time_seconds_list"].append(cpu_time_seconds)
        metrics_dict["memory_percent_list"].append(memory_percent)
        metrics_dict["memory_mb_list"].append(memory_mb)
        
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

    print()
    print("Saving performance metrics dictionary and best performing model...")

    with open(f'./outputs/metrics_dict_{file_suffix}', 'wb') as f:
        pickle.dump(metrics_dict, f)

    with open(f'./models/model_{file_suffix}', 'wb') as f:
        pickle.dump(max_assoc_mem if len(max_assoc_mem) else assoc_mem, f)

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
    print("Saving completed.")
