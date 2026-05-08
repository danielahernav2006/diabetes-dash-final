# Pestaña “Predicción” — guía de la nueva sección

Documento adjunto al README original. Describe **qué se agregó al dashboard**,
**dónde queda cada archivo** y **cómo ejecutar** la app con la nueva
funcionalidad de modelado predictivo.

---

## 1. Qué se agregó

Una nueva pestaña accesible desde el navbar (link **“Predicción”**, ruta
`/predict`) con la siguiente estructura:

1. Hero con título *“Modelo predictivo de riesgo de prediabetes/diabetes”* y
   nota académica/no diagnóstica.
2. Tarjeta destacada del **modelo recomendado** (Random Forest) con 6 KPIs:
   Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC.
3. **Selector interactivo de modelos** con descripción dinámica y métricas que
   se actualizan al cambiar la opción.
4. **Simulador de predicción individual** con 21 entradas (las mismas
   variables del entrenamiento), botón “Predecir riesgo”, panel de resultado
   con clase predicha, probabilidad y mensaje interpretativo. Permite cambiar
   el modelo usado para la simulación.
5. **Evaluación visual del mejor modelo**: matriz de confusión, curva ROC
   (todos los modelos comparados), curva Precision-Recall (todos los modelos),
   importancia de variables (top 12).
6. **Comparación general**: tabla comparativa con la fila del modelo
   recomendado destacada, gráfica de barras configurable con checklist de
   métricas, e interpretación final del desempeño.

Todo respeta la paleta y los estilos CSS existentes (`kpi-card`, `graph-card`,
`section-kicker`, `analysis-hero`, `filter-card`).

---

## 2. Archivos creados

| Archivo | Propósito |
|---|---|
| `scripts/train_models.py` | Script offline que entrena los 4 modelos y persiste artefactos en `models/`. |
| `src/utils/model_utils.py` | Carga perezosa de pipelines, métricas, importancias, mapeos UI ↔ codificación, `predict_with_model`. |
| `src/utils/model_figures.py` | Versiones Plotly de matriz de confusión, curva ROC, curva PR, importancia y comparación de métricas. Mantiene la paleta de `figures.py`. |
| `dash_app/layouts/model.py` | Layout de la pestaña `/predict`. |
| `dash_app/callbacks/model_callbacks.py` | 6 callbacks: KPIs recomendado, explorador, simulador, gráficas evaluación, tabla comparativa, gráfica comparativa. |
| `models/` | Pipelines entrenados, métricas, importancia, datos ROC/PR, matriz de confusión. |

## 3. Archivos modificados (cambios mínimos)

| Archivo | Cambio |
|---|---|
| `dash_app/index.py` | + import del nuevo layout y los callbacks · + ruta `/predict` (también acepta `/model`, `/modelo`) · + `register_model_callbacks(app)`. |
| `dash_app/components/navbar.py` | + entrada **“Predicción”** entre “Análisis” y “Equipo”. |
| `requirements.txt` | + `scikit-learn`, `xgboost`, `joblib`. |

> No se eliminó ni se renombró nada del proyecto original. Toda la
> funcionalidad existente (Inicio, Dashboard, Análisis, Equipo) sigue
> funcionando exactamente igual.

---

## 4. Cómo ejecutar

```bash
# 1) Crear entorno e instalar dependencias
pip install -r requirements.txt

# 2) (Solo la primera vez) entrenar los modelos
#    Genera todos los artefactos en models/. Tarda ~1 min en una laptop normal.
python scripts/train_models.py

# 3) Lanzar el dashboard
python dash_app/index.py
# Navegar a http://127.0.0.1:8050  →  Predicción
```

> Los archivos de `models/` ya vienen incluidos. **El paso 2 solo es
> necesario si quieres reentrenar** (por ejemplo si cambia el dataset).

---

## 5. Cómo se integró el notebook

El notebook `EDA_DIABETES_con_modelos_ML.ipynb` (sección 13) define la
pipeline completa de modelado: split estratificado, dos preprocesadores
(`preprocess_scaled` para modelos sensibles a escala, `preprocess_tree` para
ensembles), evaluación con `accuracy / precision / recall / f1 / roc_auc /
pr_auc`, y selección por **F1-score** debido al desbalance.

`scripts/train_models.py` reproduce esa lógica adaptándola para producción:

* Mantiene **el mismo split** (`test_size=0.20`, `stratify=y`,
  `random_state=42`).
* Mantiene **los mismos preprocesadores** (median + StandardScaler para
  numéricas/ordinales en LR/KNN; median puro para árboles).
* En lugar de `GridSearch/RandomizedSearch` exhaustivos (que tomaban demasiado
  para el script), fija configuraciones razonables coherentes con los rangos
  buscados en el notebook:
  - Logistic Regression: `class_weight='balanced'`, `C=1.0`.
  - KNN: `n_neighbors=51`, `weights='distance'`, entrenado sobre muestra
    estratificada de 30 000 (mismo enfoque sugerido en el notebook).
  - Random Forest: `n_estimators=250`, `max_depth=12`,
    `min_samples_leaf=5`, `class_weight='balanced'`.
  - XGBoost: `n_estimators=300`, `max_depth=4`, `lr=0.1`,
    `scale_pos_weight = neg/pos`.
* Persiste cada Pipeline completo en `models/*.joblib`. El simulador del
  dashboard llama directamente `pipeline.predict_proba`, así que el
  preprocesamiento usado en entrenamiento es **exactamente** el aplicado a
  cada predicción (cero data leakage, cero diferencia de schema).

### Métricas obtenidas

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Random Forest** | 74.24% | 31.94% | 75.07% | **44.82%** | 82.37% | 41.55% |
| XGBoost | 72.11% | 30.61% | 79.11% | 44.14% | 82.70% | 42.32% |
| Regresión Logística | 73.16% | 31.08% | 76.08% | 44.13% | 81.96% | 39.26% |
| KNN | 86.19% | 53.11% | 7.24% | 12.75% | 80.35% | 37.84% |

**Random Forest** es el modelo recomendado: mejor F1, equilibra precision y
recall, y ofrece importancia de variables interpretable. Las top 5 variables
en importancia son: `GenHlth`, `HighBP`, `BMI`, `Age`, `HighChol` —
totalmente coherente con los hallazgos del EDA.

---

## 6. Cosas que conviene revisar manualmente

1. **Compatibilidad de versiones de sklearn**: los modelos se entrenaron con
   `scikit-learn` reciente (≥1.3). Si en tu equipo tienes una versión muy
   antigua, joblib se quejará al cargarlos. Solución: re-entrenar
   (`python scripts/train_models.py`).
2. **El archivo `random_forest_pipeline.joblib` pesa ~65 MB** (250 árboles
   sobre 200 000+ filas). Si lo subes a GitHub, considera Git LFS o reducir
   `n_estimators` a 150 en `scripts/train_models.py`.
3. **XGBoost en macOS arm64** a veces requiere `brew install libomp`. Si
   `import xgboost` falla, el resto de la app sigue funcionando: la pestaña
   simplemente no mostrará XGBoost en el selector ni en la tabla.
4. **Variable de entorno `OMP_NUM_THREADS`**: si despliegas en un server con
   pocos núcleos, puede ayudar fijarla a `1` para evitar contención.
5. **Modelos sin `predict_proba`**: ya está manejado en `model_utils.py`
   (cae a `decision_function` con sigmoide; si no hay ninguno, el panel
   muestra solo la clase predicha con un mensaje explicativo). Hoy todos los
   modelos usados sí tienen `predict_proba`.
6. **Defaults del simulador**: si el usuario no toca un campo, se usa un
   valor por defecto razonable (`defaults` en `build_input_row`). Ningún
   campo puede llegar como `NaN` al pipeline porque los `SimpleImputer`
   también actuarían.
7. **No se incluyó la sección 13.13** del notebook (ajuste de umbral). Es
   una mejora opcional fácil de agregar como un slider en el panel del
   simulador. Si la quieres, dímelo y la añado.

---

## 7. Estructura final del proyecto

```
diabetes_dash_pr/
├── dash_app/
│   ├── app.py                          (sin cambios)
│   ├── index.py                        (modificado: +ruta, +callbacks)
│   ├── assets/styles.css               (sin cambios)
│   ├── components/
│   │   ├── cards.py                    (sin cambios)
│   │   ├── filters.py                  (sin cambios)
│   │   ├── kpis.py                     (sin cambios)
│   │   ├── navbar.py                   (modificado: +link Predicción)
│   │   └── tables.py                   (sin cambios)
│   ├── layouts/
│   │   ├── about.py                    (sin cambios)
│   │   ├── dashboard.py                (sin cambios)
│   │   ├── home.py                     (sin cambios)
│   │   ├── team.py                     (sin cambios)
│   │   └── model.py                    ◀ NUEVO
│   └── callbacks/
│       ├── dashboard_callbacks.py      (sin cambios)
│       ├── navigation_callbacks.py     (sin cambios)
│       └── model_callbacks.py          ◀ NUEVO
├── data/processed/diabetes_clean.csv   (sin cambios)
├── models/                             ◀ NUEVA CARPETA
│   ├── random_forest_pipeline.joblib
│   ├── xgboost_pipeline.joblib
│   ├── logistic_regression_pipeline.joblib
│   ├── knn_pipeline.joblib
│   ├── feature_columns.joblib
│   ├── confusion_matrix.joblib
│   ├── roc_pr_data.joblib
│   ├── feature_importance.csv
│   └── model_metrics.csv
├── notebooks/EDA_DIABETES.ipynb        (sin cambios)
├── scripts/
│   └── train_models.py                 ◀ NUEVO
├── src/
│   ├── data_loader.py                  (sin cambios)
│   ├── preprocessing/prepare_data.py   (sin cambios)
│   └── utils/
│       ├── figures.py                  (sin cambios)
│       ├── helpers.py                  (sin cambios)
│       ├── model_utils.py              ◀ NUEVO
│       └── model_figures.py            ◀ NUEVO
├── requirements.txt                    (modificado: +sklearn, xgboost, joblib)
└── README.md                           (sin cambios)
```
