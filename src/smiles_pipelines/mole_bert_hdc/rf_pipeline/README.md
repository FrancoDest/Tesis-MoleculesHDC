# rf_pipeline (mole_bert_rf)

Variante de [`hdc_pipeline`](../hdc_pipeline/run_hdc_pipeline.py): usa el
mismo encoder Mole-BERT para generar embeddings de grafo (300 dims, pooling
mean), pero en vez de proyectar con Johnson-Lindenstrauss y convertir a
hypervectors bipolares para un `SGDClassifier`, entrena un
`RandomForestClassifier` directo sobre el embedding de 300 dimensiones.

## Por que se salta JL + HDC

La proyeccion JL y la binarizacion a +1/-1 existen en `hdc_pipeline` porque
el clasificador final ahi es lineal (`SGDClassifier`): agrandar la dimension
(300 -> 10048) y pasar a un espacio hiperdimensional bipolar ayuda a que un
modelo lineal separe mejor las clases (es la logica clasica de HDC / Vector
Symbolic Architectures). Un Random Forest no necesita nada de eso: un
ensamble de arboles ya encuentra fronteras no lineales sobre el embedding
original sin ganar nada de la proyeccion aleatoria, asi que este pipeline la
omite directamente. Esto tambien lo hace mas simple y mas rapido por
iteracion (no hay que ajustar un proyector JL ni convertir a HDC en cada
split).

## De donde sale esto

`core.py` y `run_rf_pipeline.py` son una copia deliberada de
`hdc_pipeline/core.py` / `hdc_pipeline/run_hdc_pipeline.py` (generacion de
embeddings, carga de dataset, tracking de metricas/recursos, formato de
`metrics.json`/`test_predictions.csv`) con la proyeccion JL y la conversion
HDC eliminadas, y `SGDClassifier` reemplazado por
[`sklearn.ensemble.RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
(mismos hiperparametros que `random_forest_rdkit`: `class_weight="balanced"`,
`n_estimators` configurable). El encoder GNN en si (`../original/loader.py`,
`../original/model.py`) es el mismo codigo de Mole-BERT sin tocar, reusado
tal cual desde `hdc_pipeline`.

Se registra en `catalog.json` como `mole_bert_rf`, reusando la misma imagen
Docker que `mole_bert_hdc` (`general-molebert-hdc`, definida en
`hdc_pipeline/Dockerfile`) porque el entorno es identico -- scikit-learn ya
estaba instalado ahi para JL/evaluacion.
