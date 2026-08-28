# nn_classifier_rdkit

Baseline de red neuronal (`MLPClassifier`, perceptron multicapa de
scikit-learn) sobre descriptores RDKit, para los datasets binarios tipo
BBBP/MUTAG del proyecto (los mismos que usan `random_forest_rdkit` y
`logistic_regression_rdkit`).

## De donde sale esto

No es codigo portado de un repo externo (a diferencia de `molehd/original`,
`graphHD/original` o `mole_bert_hdc/original`). Es una implementacion propia
calcada del mismo patron que [`random_forest_rdkit`](../random_forest_rdkit/pipeline.py)
y [`logistic_regression_rdkit`](../logistic_regression_rdkit/pipeline.py):
mismos descriptores RDKit, mismas metricas y
el mismo formato de salida para integrarse con `catalog.json` y
`run_all_app.py`.

El algoritmo es la implementacion estandar de scikit-learn:
[`sklearn.neural_network.MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
(una capa oculta de 100 neuronas por defecto, configurable con
`--hidden-layer-sizes`), dentro de un `Pipeline` con
`SimpleImputer(strategy="median")` + `StandardScaler` (imprescindible para
que el solver converja). Usa `solver="adam"` (el default de sklearn) con
`early_stopping=True`, que funciono bien tanto en datasets grandes (BBBP,
~2000 filas) como en los binarizados chicos (refractive_index/glass_transition_ratio,
88-227 filas) -- a diferencia de `nn_regressor_rdkit`, donde con estos
datasets tan chicos `adam` no converge bien (ver el README de esa carpeta).
No genera un CSV de importancias/coeficientes: los pesos de
una red multicapa no se resumen en un vector por feature como `coef_` de un
modelo lineal.
