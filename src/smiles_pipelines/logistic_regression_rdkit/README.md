# logistic_regression_rdkit

Baseline de Regresion Logistica sobre descriptores RDKit, para los datasets
binarios tipo BBBP/MUTAG del proyecto (los mismos que usa
`random_forest_rdkit`).

## De donde sale esto

Igual que [`linear_regression_rdkit`](../linear_regression_rdkit/README.md),
no es codigo portado de un repo externo (a diferencia de `molehd/original`,
`graphHD/original` o `mole_bert_hdc/original`). Es una implementacion propia
calcada de [`src/random_forest_rdkit/pipeline.py`](../random_forest_rdkit/pipeline.py):
mismos descriptores RDKit, mismas metricas
(accuracy/balanced_accuracy/precision/recall/f1/roc_auc/confusion_matrix) y
el mismo formato de salida (`metrics.json`, `detailed_results.csv`,
`test_predictions.csv`) para integrarse con `catalog.json` y `run_all_app.py`.

El algoritmo es la implementacion estandar de scikit-learn:
[`sklearn.linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
(regularizacion L2 por defecto, `class_weight="balanced"` para compensar el
desbalance de clases igual que el Random Forest), dentro de un `Pipeline` con
`SimpleImputer(strategy="median")` + `StandardScaler`. El escalado es
necesario para que el solver (`lbfgs`) converja razonablemente rapido dado
que los descriptores RDKit tienen escalas muy distintas entre si.
