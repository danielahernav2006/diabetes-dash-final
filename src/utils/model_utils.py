"""
Utilidades para la pestaña de modelado predictivo.

- Carga perezosa de los pipelines entrenados (`models/*.joblib`)
- Carga de métricas, importancias, datos para curvas ROC / PR y matriz de confusión
- Mapeos UI ↔ codificación numérica del dataset BRFSS 2015 para el simulador
- Construcción del DataFrame de entrada para `pipeline.predict_proba`
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# joblib es la única dependencia "pesada" estrictamente necesaria en runtime
# para cargar los modelos. Si los archivos no existen, los helpers retornan
# None / DataFrames vacíos para que la app siga corriendo.
try:
    import joblib
    _JOBLIB_OK = True
except Exception:
    _JOBLIB_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# Rutas
# ──────────────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_SRC)
MODELS_DIR = os.path.join(_ROOT, "models")


# ──────────────────────────────────────────────────────────────────────────────
# Catálogo de modelos
# ──────────────────────────────────────────────────────────────────────────────
MODEL_FILES = {
    "Random Forest":        "random_forest_pipeline.joblib",
    "XGBoost":              "xgboost_pipeline.joblib",
    "Logistic Regression":  "logistic_regression_pipeline.joblib",
    "KNN":                  "knn_pipeline.joblib",
}

# Etiqueta visible en español para cada nombre canónico
MODEL_LABELS_ES = {
    "Random Forest":        "Random Forest",
    "XGBoost":              "XGBoost",
    "Logistic Regression":  "Regresión Logística",
    "KNN":                  "KNN",
}

# Descripciones académicas breves
MODEL_DESCRIPTIONS = {
    "Random Forest": (
        "Random Forest es un modelo basado en múltiples árboles de decisión. "
        "Su fortaleza es capturar relaciones no lineales entre variables y "
        "reducir la varianza al combinar varios árboles. En este proyecto fue "
        "el modelo con mejor F1-score, equilibrando precision y recall en un "
        "contexto de clases desbalanceadas."
    ),
    "XGBoost": (
        "XGBoost es un modelo de boosting basado en árboles de decisión. "
        "Construye los árboles de forma secuencial corrigiendo errores previos. "
        "Suele desempeñarse muy bien en datos tabulares; aquí se usó "
        "`scale_pos_weight` para manejar el desbalance de clases."
    ),
    "Logistic Regression": (
        "La Regresión Logística es un modelo lineal utilizado para "
        "clasificación binaria. Aunque es más simple e interpretable, sirve "
        "como modelo base sólido para comparar contra algoritmos más "
        "complejos. Con `class_weight='balanced'` mejora su recall."
    ),
    "KNN": (
        "KNN clasifica cada observación según la clase predominante entre sus "
        "vecinos más cercanos. En este caso obtuvo accuracy alto pero recall "
        "muy bajo, por lo que no es adecuado para detectar correctamente la "
        "clase positiva en este dataset desbalanceado."
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Caché de modelos / artefactos
# ──────────────────────────────────────────────────────────────────────────────
_MODEL_CACHE: Dict[str, object] = {}
_METRICS_DF: Optional[pd.DataFrame] = None
_FEATURE_IMPORTANCE_DF: Optional[pd.DataFrame] = None
_ROC_PR_DATA: Optional[dict] = None
_CONF_MATRIX: Optional[dict] = None
_FEATURE_COLUMNS: Optional[list] = None


def _safe_load(path):
    if not _JOBLIB_OK:
        return None
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[model_utils] Aviso: no pude cargar {path}: {e}")
        return None


def get_model(name: str):
    """Carga perezosa de un pipeline entrenado. Devuelve None si no existe."""
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]
    fname = MODEL_FILES.get(name)
    if not fname:
        return None
    model = _safe_load(os.path.join(MODELS_DIR, fname))
    if model is not None:
        _MODEL_CACHE[name] = model
    return model


def get_metrics_df() -> pd.DataFrame:
    """Devuelve la tabla de métricas (model_metrics.csv) ordenada por F1 desc."""
    global _METRICS_DF
    if _METRICS_DF is not None:
        return _METRICS_DF.copy()
    path = os.path.join(MODELS_DIR, "model_metrics.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=[
            "model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
        ])
    df = df.sort_values("f1", ascending=False).reset_index(drop=True)
    _METRICS_DF = df
    return df.copy()


def get_best_model_name() -> str:
    df = get_metrics_df()
    if df.empty:
        return "Random Forest"
    return str(df.iloc[0]["model"])


def get_feature_importance() -> pd.DataFrame:
    global _FEATURE_IMPORTANCE_DF
    if _FEATURE_IMPORTANCE_DF is not None:
        return _FEATURE_IMPORTANCE_DF.copy()
    path = os.path.join(MODELS_DIR, "feature_importance.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["feature", "importance", "model"])
    _FEATURE_IMPORTANCE_DF = df
    return df.copy()


def get_roc_pr_data() -> dict:
    global _ROC_PR_DATA
    if _ROC_PR_DATA is not None:
        return _ROC_PR_DATA
    data = _safe_load(os.path.join(MODELS_DIR, "roc_pr_data.joblib")) or {}
    _ROC_PR_DATA = data
    return data


def get_confusion_matrix() -> Optional[dict]:
    global _CONF_MATRIX
    if _CONF_MATRIX is not None:
        return _CONF_MATRIX
    data = _safe_load(os.path.join(MODELS_DIR, "confusion_matrix.joblib"))
    _CONF_MATRIX = data
    return data


def get_feature_columns() -> list:
    """Orden de columnas usado en entrenamiento. Imprescindible para construir
    el DataFrame de entrada del simulador con el mismo schema."""
    global _FEATURE_COLUMNS
    if _FEATURE_COLUMNS is not None:
        return list(_FEATURE_COLUMNS)
    cols = _safe_load(os.path.join(MODELS_DIR, "feature_columns.joblib"))
    if cols is None:
        # Fallback al orden conocido del dataset BRFSS 2015 (sin la target)
        cols = [
            "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
            "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
            "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
            "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
            "Sex", "Age", "Education", "Income",
        ]
    _FEATURE_COLUMNS = list(cols)
    return list(_FEATURE_COLUMNS)


# ──────────────────────────────────────────────────────────────────────────────
# Mapeos UI → códigos del dataset (idénticos a data_loader.py)
# ──────────────────────────────────────────────────────────────────────────────
BINARY_OPTIONS = [
    {"label": "No", "value": 0},
    {"label": "Sí", "value": 1},
]

SEX_OPTIONS = [
    {"label": "Mujer",  "value": 0},
    {"label": "Hombre", "value": 1},
]

GENHLTH_OPTIONS = [
    {"label": "Excelente",   "value": 1},
    {"label": "Muy buena",   "value": 2},
    {"label": "Buena",       "value": 3},
    {"label": "Regular",     "value": 4},
    {"label": "Mala",        "value": 5},
]

AGE_OPTIONS = [
    {"label": "18-24",  "value": 1},
    {"label": "25-29",  "value": 2},
    {"label": "30-34",  "value": 3},
    {"label": "35-39",  "value": 4},
    {"label": "40-44",  "value": 5},
    {"label": "45-49",  "value": 6},
    {"label": "50-54",  "value": 7},
    {"label": "55-59",  "value": 8},
    {"label": "60-64",  "value": 9},
    {"label": "65-69",  "value": 10},
    {"label": "70-74",  "value": 11},
    {"label": "75-79",  "value": 12},
    {"label": "80+",    "value": 13},
]

EDUCATION_OPTIONS = [
    {"label": "Nunca asistió",          "value": 1},
    {"label": "Primaria incompleta",    "value": 2},
    {"label": "Secundaria incompleta",  "value": 3},
    {"label": "Secundaria completa / GED", "value": 4},
    {"label": "Universidad incompleta", "value": 5},
    {"label": "Universidad completa",   "value": 6},
]

INCOME_OPTIONS = [
    {"label": "< $10.000",        "value": 1},
    {"label": "$10.000 - $15.000", "value": 2},
    {"label": "$15.000 - $20.000", "value": 3},
    {"label": "$20.000 - $25.000", "value": 4},
    {"label": "$25.000 - $35.000", "value": 5},
    {"label": "$35.000 - $50.000", "value": 6},
    {"label": "$50.000 - $75.000", "value": 7},
    {"label": "$75.000 o más",     "value": 8},
]

# Etiquetas largas y nombres "humanos" de variables (para tooltips, importancia)
FEATURE_LABELS_ES = {
    "HighBP":               "Hipertensión",
    "HighChol":             "Colesterol alto",
    "CholCheck":            "Chequeo de colesterol",
    "BMI":                  "IMC (Índice de masa corporal)",
    "Smoker":               "Fumador",
    "Stroke":               "Accidente cerebrovascular previo",
    "HeartDiseaseorAttack": "Enfermedad cardíaca o infarto",
    "PhysActivity":         "Actividad física",
    "Fruits":               "Consumo de frutas",
    "Veggies":              "Consumo de verduras",
    "HvyAlcoholConsump":    "Consumo elevado de alcohol",
    "AnyHealthcare":        "Cobertura médica",
    "NoDocbcCost":          "Barrera económica de acceso",
    "GenHlth":              "Salud general percibida",
    "MentHlth":             "Días con mala salud mental",
    "PhysHlth":             "Días con mala salud física",
    "DiffWalk":             "Dificultad para caminar",
    "Sex":                  "Sexo",
    "Age":                  "Grupo de edad",
    "Education":            "Nivel educativo",
    "Income":               "Nivel de ingreso",
}


# ──────────────────────────────────────────────────────────────────────────────
# Construcción del DataFrame de entrada para el pipeline
# ──────────────────────────────────────────────────────────────────────────────

def build_input_row(form_values: dict) -> pd.DataFrame:
    """Construye un DataFrame de una fila con TODAS las columnas del entrenamiento.

    Parameters
    ----------
    form_values : dict
        Diccionario {nombre_columna: valor}. Si falta alguna columna se rellena
        con un valor por defecto razonable (la mediana / el valor más común
        para variables binarias) — pero el imputador del Pipeline también
        manejará faltantes si llegasen como NaN.
    """
    cols = get_feature_columns()

    # Defaults seguros (basados en valores comunes del dataset BRFSS 2015)
    defaults = {
        "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 27.0, "Smoker": 0,
        "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1,
        "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 3,
        "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": 0,
        "Age": 9, "Education": 5, "Income": 6,
    }

    row = {}
    for c in cols:
        v = form_values.get(c)
        if v is None or v == "":
            row[c] = defaults.get(c, 0)
        else:
            row[c] = v

    return pd.DataFrame([row], columns=cols)


def predict_with_model(model_name: str, form_values: dict) -> Tuple[Optional[int], Optional[float]]:
    """Predice clase y probabilidad usando el modelo solicitado.

    Returns
    -------
    (label, prob) : (int o None, float o None)
        - label: 0 o 1; None si el modelo no está disponible.
        - prob:  probabilidad estimada de la clase positiva; None si el modelo
          no tiene predict_proba ni decision_function.
    """
    model = get_model(model_name)
    if model is None:
        return None, None

    X_in = build_input_row(form_values)

    try:
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_in)[0, 1])
            label = int(proba >= 0.5)
            return label, proba
        if hasattr(model, "decision_function"):
            score = float(model.decision_function(X_in)[0])
            # Aproximación a probabilidad mediante sigmoide
            proba = 1.0 / (1.0 + np.exp(-score))
            label = int(proba >= 0.5)
            return label, proba
        # Sin probabilidades disponibles
        label = int(model.predict(X_in)[0])
        return label, None
    except Exception as e:
        print(f"[model_utils] Error prediciendo con {model_name}: {e}")
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de formato
# ──────────────────────────────────────────────────────────────────────────────

def fmt_pct(value, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:.{decimals}f}%"
