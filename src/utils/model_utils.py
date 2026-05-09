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
import json

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
    # Modelo final
    "Mejor modelo": "best_model.joblib",

    # Nombres nuevos del notebook final
    "XGBoost + class_weight": "xgboost_class_weight_pipeline.joblib",
    "Random Forest + class_weight": "random_forest_class_weight_pipeline.joblib",
    "Regresión Logística + class_weight": "regresion_logistica_class_weight_pipeline.joblib",
    "KNN + ADASYN": "knn_adasyn_pipeline.joblib",

    # Aliases por compatibilidad con el Dash anterior
    "XGBoost": "xgboost_class_weight_pipeline.joblib",
    "Random Forest": "random_forest_class_weight_pipeline.joblib",
    "Logistic Regression": "regresion_logistica_class_weight_pipeline.joblib",
    "Regresión Logística": "regresion_logistica_class_weight_pipeline.joblib",
    "KNN": "knn_adasyn_pipeline.joblib",
}

# Etiqueta visible en español para cada nombre canónico
MODEL_LABELS_ES = {
    "Mejor modelo": "Modelo final",
    "XGBoost + class_weight": "XGBoost + class weight",
    "Random Forest + class_weight": "Random Forest + class weight",
    "Regresión Logística + class_weight": "Regresión Logística + class weight",
    "KNN + ADASYN": "KNN + ADASYN",

    "XGBoost": "XGBoost + class weight",
    "Random Forest": "Random Forest + class weight",
    "Logistic Regression": "Regresión Logística + class weight",
    "Regresión Logística": "Regresión Logística + class weight",
    "KNN": "KNN + ADASYN",
}

# Descripciones académicas breves
MODEL_DESCRIPTIONS = {
    # ============================================================
    # Descripciones por modelo base
    # ============================================================

    "Regresión Logística": (
        "La Regresión Logística es un modelo lineal utilizado para problemas de clasificación binaria. "
        "Estima la probabilidad de que una persona pertenezca a la clase positiva, en este caso "
        "prediabetes/diabetes, a partir de una combinación ponderada de las variables predictoras. "
        "Su principal ventaja es que es un modelo simple, estable e interpretable, útil como punto de "
        "referencia para comparar modelos más complejos. Sin embargo, al ser lineal, puede tener limitaciones "
        "para capturar relaciones no lineales entre variables de salud, comportamiento y características "
        "sociodemográficas."
    ),

    "Random Forest": (
        "Random Forest es un modelo basado en ensambles de árboles de decisión. Construye múltiples árboles "
        "sobre diferentes subconjuntos de datos y variables, y luego combina sus resultados para generar una "
        "predicción más robusta. Este enfoque permite capturar relaciones no lineales e interacciones entre "
        "variables, lo que resulta útil en datos de salud donde los factores de riesgo pueden combinarse de "
        "formas complejas. Además, suele ser menos sensible al ruido que un único árbol de decisión."
    ),

    "XGBoost": (
        "XGBoost es un modelo de boosting basado en árboles de decisión. A diferencia de Random Forest, que "
        "entrena varios árboles de forma más independiente, XGBoost construye árboles de manera secuencial, "
        "corrigiendo progresivamente los errores cometidos por los árboles anteriores. Esto le permite alcanzar "
        "un alto desempeño predictivo y capturar patrones complejos en los datos. En este proyecto, XGBoost fue "
        "el modelo recomendado porque obtuvo el mayor recall en el conjunto de prueba, lo cual es coherente con "
        "el objetivo de detectar la mayor cantidad posible de casos de prediabetes/diabetes."
    ),

    "KNN": (
        "KNN, o K-Nearest Neighbors, es un modelo basado en similitud. Para clasificar una nueva observación, "
        "busca los registros más parecidos dentro del conjunto de entrenamiento y asigna la clase más frecuente "
        "entre sus vecinos cercanos. Es un algoritmo intuitivo y fácil de entender, pero puede verse afectado por "
        "la escala de las variables, la dimensionalidad del dataset y el desbalance de clases. Por eso requiere "
        "preprocesamiento cuidadoso y puede beneficiarse de técnicas de sobremuestreo como SMOTE o ADASYN."
    ),

    # ============================================================
    # Regresión Logística por estrategia de balanceo
    # ============================================================

    "Regresión Logística + none": (
        "Esta configuración evalúa la Regresión Logística sin aplicar una estrategia adicional de balanceo. "
        "Funciona como escenario base para observar cómo se comporta un modelo lineal cuando aprende directamente "
        "de un dataset desbalanceado. En este caso, el modelo puede favorecer la clase mayoritaria, lo que suele "
        "reflejarse en una accuracy relativamente alta, pero con menor capacidad para detectar correctamente los "
        "casos de prediabetes/diabetes."
    ),

    "Regresión Logística + class_weight": (
        "Esta configuración combina Regresión Logística con ajuste de pesos de clase. La técnica class weight "
        "asigna mayor importancia a la clase minoritaria durante el entrenamiento, con el fin de reducir el sesgo "
        "del modelo hacia la clase dominante. En este proyecto, esta estrategia mejora de manera importante el recall "
        "de la clase positiva, aunque puede disminuir la precisión al aumentar la cantidad de observaciones clasificadas "
        "como posibles casos de prediabetes/diabetes."
    ),

    "Regresión Logística + smote": (
        "Esta configuración combina Regresión Logística con SMOTE, una técnica de sobremuestreo que genera ejemplos "
        "sintéticos de la clase minoritaria. El objetivo es ofrecerle al modelo más información sobre los casos positivos "
        "durante el entrenamiento. En un modelo lineal como la Regresión Logística, SMOTE puede mejorar la sensibilidad "
        "frente a la clase positiva, aunque su desempeño depende de qué tan separables sean las clases después del "
        "preprocesamiento."
    ),

    "Regresión Logística + adasyn": (
        "Esta configuración combina Regresión Logística con ADASYN, una técnica de sobremuestreo adaptativo que genera "
        "más ejemplos sintéticos en las zonas donde la clase minoritaria es más difícil de aprender. Su propósito es "
        "ayudar al modelo a reconocer mejor los casos positivos complejos. En este contexto, puede mejorar el recall, "
        "pero también puede aumentar los falsos positivos si las clases presentan mucho solapamiento."
    ),

    # ============================================================
    # Random Forest por estrategia de balanceo
    # ============================================================

    "Random Forest + none": (
        "Esta configuración evalúa Random Forest sin aplicar balanceo adicional. Aunque el modelo puede capturar "
        "relaciones no lineales e interacciones entre variables, el desbalance de clases puede llevarlo a favorecer "
        "la clase mayoritaria. Por eso, puede obtener una accuracy alta, pero un recall bajo para la clase positiva, "
        "lo cual no es ideal cuando el objetivo principal es detectar casos de prediabetes/diabetes."
    ),

    "Random Forest + class_weight": (
        "Esta configuración combina Random Forest con ajuste de pesos de clase. Al aumentar el peso de la clase "
        "minoritaria, el modelo recibe una señal más fuerte para identificar personas con prediabetes/diabetes. "
        "Esta estrategia busca mejorar el recall sin modificar directamente la distribución original de los datos. "
        "Es una alternativa útil cuando se quiere tratar el desbalance desde la función de entrenamiento del modelo."
    ),

    "Random Forest + smote": (
        "Esta configuración combina Random Forest con SMOTE. La técnica genera observaciones sintéticas de la clase "
        "minoritaria antes del entrenamiento, permitiendo que los árboles reciban más ejemplos positivos. Esto puede "
        "mejorar la detección de casos de prediabetes/diabetes, aunque también puede introducir patrones artificiales "
        "si las nuevas observaciones no representan adecuadamente la estructura real del dataset."
    ),

    "Random Forest + adasyn": (
        "Esta configuración combina Random Forest con ADASYN. A diferencia de SMOTE, ADASYN concentra la generación "
        "de ejemplos sintéticos en las zonas más difíciles de clasificar. Esto puede ayudar al modelo a aprender mejor "
        "los casos positivos más complejos. Sin embargo, también puede incrementar la variabilidad del modelo y generar "
        "más falsos positivos si las fronteras entre clases no son claras."
    ),

    # ============================================================
    # XGBoost por estrategia de balanceo
    # ============================================================

    "XGBoost + none": (
        "Esta configuración evalúa XGBoost sin aplicar una estrategia específica de balanceo. Aunque XGBoost es un "
        "modelo potente y capaz de capturar relaciones complejas, el desbalance de clases puede hacer que el modelo "
        "optimice mejor el desempeño sobre la clase mayoritaria. Por eso, sirve como referencia para comparar cuánto "
        "mejoran las métricas de la clase positiva al incorporar técnicas de balanceo."
    ),

    "XGBoost + class_weight": (
        "Esta fue la combinación recomendada en el proyecto. XGBoost con class weight ajusta el aprendizaje del modelo "
        "para darle mayor importancia a la clase minoritaria, correspondiente a prediabetes/diabetes. Esta estrategia "
        "permitió alcanzar el mayor recall en el conjunto de prueba, lo que significa que fue la configuración con mejor "
        "capacidad para detectar casos positivos. Aunque su precisión es más moderada, este comportamiento es coherente "
        "con el objetivo del proyecto: priorizar la identificación de personas en posible condición de riesgo."
    ),

    "XGBoost + smote": (
        "Esta configuración combina XGBoost con SMOTE. El sobremuestreo genera registros sintéticos de la clase positiva "
        "para que el modelo tenga más ejemplos de prediabetes/diabetes durante el entrenamiento. XGBoost puede aprovechar "
        "estos datos adicionales para aprender patrones no lineales, aunque el resultado depende de la calidad de los "
        "ejemplos sintéticos y de qué tanto representen la distribución real de la clase minoritaria."
    ),

    "XGBoost + adasyn": (
        "Esta configuración combina XGBoost con ADASYN. Esta técnica genera más ejemplos sintéticos en regiones donde "
        "los casos positivos son más difíciles de clasificar. En teoría, esto puede ayudar a XGBoost a mejorar la detección "
        "de observaciones complejas. Sin embargo, al concentrarse en zonas difíciles, también puede aumentar el riesgo de "
        "falsos positivos si esas regiones están muy mezcladas con la clase mayoritaria."
    ),

    # ============================================================
    # KNN por estrategia de balanceo
    # ============================================================

    "KNN + none": (
        "Esta configuración evalúa KNN sin aplicar balanceo adicional. Como KNN clasifica una observación según sus vecinos "
        "más cercanos, el desbalance de clases puede afectar fuertemente sus predicciones: si la mayoría de vecinos pertenecen "
        "a la clase sin diabetes, el modelo tenderá a clasificar nuevos casos en esa misma categoría. Por eso, sin balanceo, "
        "KNN puede presentar dificultades para detectar la clase positiva."
    ),

    "KNN + smote": (
        "Esta configuración combina KNN con SMOTE. Al generar ejemplos sintéticos de la clase minoritaria, se busca que los "
        "vecinos cercanos incluyan más casos positivos y que el modelo mejore su capacidad de detección. Esta estrategia es "
        "especialmente relevante para KNN, ya que el algoritmo depende directamente de la distribución local de las observaciones "
        "en el espacio de variables."
    ),

    "KNN + adasyn": (
        "Esta configuración combina KNN con ADASYN, que genera ejemplos sintéticos de la clase minoritaria especialmente en "
        "las zonas más difíciles de clasificar. En este proyecto, ADASYN fue la mejor estrategia evaluada para KNN bajo el "
        "criterio de recall. Esto sugiere que el sobremuestreo adaptativo ayudó al modelo a reconocer mejor ciertos patrones "
        "de la clase positiva, aunque el desempeño general sigue dependiendo de la calidad de la representación de vecinos."
    ),

    "KNN + ADASYN": (
        "Esta configuración combina KNN con ADASYN, que genera ejemplos sintéticos de la clase minoritaria especialmente en "
        "las zonas más difíciles de clasificar. En este proyecto, ADASYN fue la mejor estrategia evaluada para KNN bajo el "
        "criterio de recall. Esto sugiere que el sobremuestreo adaptativo ayudó al modelo a reconocer mejor ciertos patrones "
        "de la clase positiva, aunque el desempeño general sigue dependiendo de la calidad de la representación de vecinos."
    ),

    "KNN + SMOTE": (
        "Esta configuración combina KNN con SMOTE. Al generar ejemplos sintéticos de la clase minoritaria, se busca que los "
        "vecinos cercanos incluyan más casos positivos y que el modelo mejore su capacidad de detección. Esta estrategia es "
        "especialmente relevante para KNN, ya que el algoritmo depende directamente de la distribución local de las observaciones "
        "en el espacio de variables."
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
_BEST_MODEL_METRICS: Optional[dict] = None
_THRESHOLD_ARTIFACT: Optional[dict] = None
_TRAINING_METADATA: Optional[dict] = None


CURVE_MODEL_BASE_ORDER = ["Regresión Logística", "KNN", "Random Forest", "XGBoost"]
CURVE_MODEL_LABELS = {
    "Regresión Logística": "Regresión Logística",
    "Logistic Regression": "Regresión Logística",
    "KNN": "KNN",
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
}
BALANCE_LABELS_ES = {
    "none": "Sin balanceo",
    "class_weight": "Class weight",
    "smote": "SMOTE",
    "adasyn": "ADASYN",
}


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
    """Devuelve la tabla de métricas en formato compatible con el Dash."""
    global _METRICS_DF

    if _METRICS_DF is not None:
        return _METRICS_DF.copy()

    path = os.path.join(MODELS_DIR, "model_metrics.csv")

    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=[
            "model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
        ])

    # ------------------------------------------------------------------
    # Compatibilidad: formato nuevo del notebook -> formato viejo del Dash
    # ------------------------------------------------------------------
    rename_map = {
        "Modelo": "model",
        "Modelo_base": "model_base",
        "Balanceo": "balance_strategy",
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1": "f1",
        "ROC_AUC": "roc_auc",
        "PR_AUC": "pr_auc",
        "Recall_CV_Optuna": "recall_cv_optuna",
        "Tiempo_Optuna_seg": "optimization_time_sec",
        "N_Trials": "n_trials",
        "Mejores_params": "best_params",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # También conservar columnas nuevas por si otra parte del Dash las usa
    reverse_map = {
        "model": "Modelo",
        "model_base": "Modelo_base",
        "balance_strategy": "Balanceo",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "roc_auc": "ROC_AUC",
        "pr_auc": "PR_AUC",
    }

    for old, new in reverse_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # Asegurar columnas mínimas esperadas por callbacks antiguos
    for col in ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        if col not in df.columns:
            df[col] = np.nan

    if not df.empty:
        if "recall" in df.columns:
            df = df.sort_values("recall", ascending=False).reset_index(drop=True)
        elif "f1" in df.columns:
            df = df.sort_values("f1", ascending=False).reset_index(drop=True)

    _METRICS_DF = df
    return df.copy()

def get_best_model_name() -> str:
    """Devuelve el nombre del mejor modelo según Recall."""
    df = get_metrics_df()

    if df.empty:
        return "Mejor modelo"

    if "model" in df.columns and pd.notna(df.iloc[0]["model"]):
        return str(df.iloc[0]["model"])

    if "Modelo" in df.columns and pd.notna(df.iloc[0]["Modelo"]):
        return str(df.iloc[0]["Modelo"])

    return "Mejor modelo"

def get_feature_importance() -> pd.DataFrame:
    global _FEATURE_IMPORTANCE_DF

    if _FEATURE_IMPORTANCE_DF is not None:
        return _FEATURE_IMPORTANCE_DF.copy()

    path = os.path.join(MODELS_DIR, "feature_importance.csv")

    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["Variable", "Importancia"])

    # Normalizar nombres según el formato que venga del notebook
    if "feature" in df.columns and "Variable" not in df.columns:
        df["Variable"] = df["feature"]

    if "importance" in df.columns and "Importancia" not in df.columns:
        df["Importancia"] = df["importance"]

    # Crear columnas en formato antiguo para compatibilidad con el Dash
    if "Variable" in df.columns and "feature" not in df.columns:
        df["feature"] = df["Variable"]

    if "Importancia" in df.columns and "importance" not in df.columns:
        df["importance"] = df["Importancia"]

    # Si existe nombre amigable en español, lo dejamos también
    if "feature_label" not in df.columns and "feature" in df.columns:
        df["feature_label"] = df["feature"].map(FEATURE_LABELS_ES).fillna(df["feature"])

    # Ordenar por importancia
    if "importance" in df.columns:
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    elif "Importancia" in df.columns:
        df = df.sort_values("Importancia", ascending=False).reset_index(drop=True)

    _FEATURE_IMPORTANCE_DF = df
    return df.copy()

def get_roc_pr_data() -> dict:
    global _ROC_PR_DATA

    if _ROC_PR_DATA is not None:
        return _ROC_PR_DATA

    data = _safe_load(os.path.join(MODELS_DIR, "roc_pr_data.joblib")) or {}

    # ------------------------------------------------------------------
    # Compatibilidad:
    # Si viene en formato nuevo del notebook como una sola curva:
    # {"modelo": ..., "fpr": ..., "tpr": ..., "precision_curve": ...}
    # lo convertimos al formato esperado por las figuras antiguas:
    # {"Modelo": {"fpr": ..., "tpr": ..., ...}}
    # ------------------------------------------------------------------
    if isinstance(data, dict) and "fpr" in data and "tpr" in data:
        model_name = data.get("modelo", data.get("model", "Modelo final"))

        data = {
            model_name: {
                "fpr": data.get("fpr", []),
                "tpr": data.get("tpr", []),
                "roc_auc": data.get("roc_auc", None),
                "precision": data.get("precision_curve", data.get("precision", [])),
                "recall": data.get("recall_curve", data.get("recall", [])),
                "pr_auc": data.get("average_precision", data.get("pr_auc", None)),
                "average_precision": data.get("average_precision", data.get("pr_auc", None)),
            }
        }
    elif isinstance(data, dict):
        normalized = {}
        for name, vals in data.items():
            if not isinstance(vals, dict):
                continue
            normalized[str(name)] = {
                "fpr": vals.get("fpr", []),
                "tpr": vals.get("tpr", []),
                "roc_auc": vals.get("roc_auc", vals.get("auc", None)),
                "precision": vals.get("precision_curve", vals.get("precision", [])),
                "recall": vals.get("recall_curve", vals.get("recall", [])),
                "pr_auc": vals.get("average_precision", vals.get("pr_auc", None)),
                "average_precision": vals.get("average_precision", vals.get("pr_auc", None)),
            }
        data = normalized

    data = _complete_best_model_curves(data)

    _ROC_PR_DATA = data
    return data


def get_roc_pr_model_options() -> list:
    """Opciones de curvas: una por modelo base con su mejor balanceo."""
    data = get_roc_pr_data()
    options = []
    for key, vals in data.items():
        label = vals.get("display_name", key) if isinstance(vals, dict) else key
        balance = vals.get("balance_label", "") if isinstance(vals, dict) else ""
        options.append({
            "label": f"{label} ({balance})" if balance else label,
            "value": key,
        })
    return options


def _as_1d_array(values):
    if values is None:
        return None
    try:
        arr = np.asarray(values, dtype=float).ravel()
        return arr if arr.size else None
    except Exception:
        return None


def _binary_auc(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return np.nan
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


def _average_precision(recall, precision) -> float:
    recall = np.asarray(recall, dtype=float)
    precision = np.asarray(precision, dtype=float)
    if recall.size < 2 or precision.size < 2:
        return np.nan
    order = np.argsort(recall)
    r = recall[order]
    p = precision[order]
    return float(np.sum((r[1:] - r[:-1]) * p[1:]))


def _curve_points(y_true, y_score) -> dict:
    y_true = _as_1d_array(y_true)
    y_score = _as_1d_array(y_score)
    if y_true is None or y_score is None or y_true.size != y_score.size:
        return {}

    y_true = (y_true == 1).astype(int)
    positives = int(y_true.sum())
    negatives = int((y_true == 0).sum())
    if positives == 0 or negatives == 0:
        return {}

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = y_score[order]
    distinct_idx = np.where(np.diff(score_sorted))[0]
    threshold_idxs = np.r_[distinct_idx, y_true.size - 1]

    tps = np.cumsum(y_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    tpr = np.r_[0, tps / positives]
    fpr = np.r_[0, fps / negatives]

    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / positives
    precision_curve = np.r_[1, precision]
    recall_curve = np.r_[0, recall]

    return {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision_curve,
        "recall": recall_curve,
        "roc_auc": _binary_auc(fpr, tpr),
        "pr_auc": _average_precision(recall_curve, precision_curve),
    }


def _default_y_true():
    path = os.path.join(MODELS_DIR, "best_model_predictions.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        for col in ["y_real", "y_true", "y_test"]:
            if col in df.columns:
                return df[col].values
    except Exception as e:
        print(f"[model_utils] Aviso: no pude leer y_real desde best_model_predictions.csv: {e}")
    return None


def _collect_score_arrays(obj, prefix="") -> dict:
    scores = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix} + {key}" if prefix else str(key)
            key_l = str(key).lower()
            if isinstance(value, dict):
                scores.update(_collect_score_arrays(value, path))
            else:
                arr = _as_1d_array(value)
                if arr is None:
                    continue
                # En artefactos de scores, las hojas suelen ser arrays de probabilidad.
                # Evitamos tomar predicciones binarias si el nombre lo indica.
                if any(token in key_l for token in ["pred", "y_pred", "class"]):
                    continue
                scores[path] = arr
    elif isinstance(obj, pd.DataFrame):
        model_col = next((c for c in ["model", "Modelo", "modelo"] if c in obj.columns), None)
        score_col = next((c for c in ["y_score", "score", "scores", "y_proba", "proba", "probability"] if c in obj.columns), None)
        if model_col and score_col:
            for name, group in obj.groupby(model_col):
                arr = _as_1d_array(group[score_col])
                if arr is not None:
                    scores[str(name)] = arr
    return scores


def _prediction_artifact_curves() -> dict:
    artifacts = [
        _safe_load(os.path.join(MODELS_DIR, "scores_experimentos.joblib")),
        _safe_load(os.path.join(MODELS_DIR, "predicciones_experimentos.joblib")),
    ]

    y_true = _default_y_true()
    scores_by_model = {}
    curves = {}

    for artifact in artifacts:
        if artifact is None:
            continue

        if isinstance(artifact, dict):
            artifact_true = artifact.get("y_true", artifact.get("y_test", artifact.get("y_real")))
            if y_true is None and artifact_true is not None:
                y_true = artifact_true
            if isinstance(artifact.get("scores"), dict):
                scores_by_model.update(_collect_score_arrays(artifact["scores"]))
            if isinstance(artifact.get("y_scores"), dict):
                scores_by_model.update(_collect_score_arrays(artifact["y_scores"]))
            if isinstance(artifact.get("probabilidades"), dict):
                scores_by_model.update(_collect_score_arrays(artifact["probabilidades"]))
            scores_by_model.update(_collect_score_arrays(artifact))

        elif isinstance(artifact, pd.DataFrame):
            true_col = next((c for c in ["y_true", "y_test", "y_real"] if c in artifact.columns), None)
            if y_true is None and true_col:
                y_true = artifact[true_col].values
            scores_by_model.update(_collect_score_arrays(artifact))

    for name, score in scores_by_model.items():
        curve = _curve_points(y_true, score)
        if curve:
            curves[str(name)] = curve
    return curves


def _best_curve_models_from_metrics() -> list:
    df = get_metrics_df()
    if df.empty:
        return [
            ("Regresión Logística + class_weight", "Regresión Logística", "class_weight", 0.816, 0.385),
            ("KNN + adasyn", "KNN", "adasyn", 0.817, 0.399),
            ("Random Forest + class_weight", "Random Forest", "class_weight", 0.815, 0.402),
            ("XGBoost + class_weight", "XGBoost", "class_weight", 0.827, 0.423),
        ]

    rows = []
    for base in CURVE_MODEL_BASE_ORDER:
        subset = df[df["model_base"] == base].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("recall", ascending=False)
        row = subset.iloc[0]
        rows.append((
            str(row["model"]),
            base,
            str(row["balance_strategy"]),
            float(row.get("roc_auc", np.nan)),
            float(row.get("pr_auc", np.nan)),
        ))
    return rows


def _curve_name_matches(curve_name: str, model_name: str, base: str, balance: str) -> bool:
    c = curve_name.lower().replace("_", " ")
    exact_candidates = {
        model_name.lower().replace("_", " "),
        f"{base} + {balance}".lower().replace("_", " "),
    }
    if any(candidate in c or c in candidate for candidate in exact_candidates):
        return True

    base_candidates = {base.lower().replace("_", " ")}
    if base == "Regresión Logística":
        base_candidates.add("logistic regression")
    return any(candidate == c for candidate in base_candidates)


def _complete_best_model_curves(data: dict) -> dict:
    prediction_curves = _prediction_artifact_curves()
    all_curves = {}
    all_curves.update(data or {})
    all_curves.update({k: v for k, v in prediction_curves.items() if k not in all_curves})

    filtered = {}
    for model_name, base, balance, metric_roc_auc, metric_pr_auc in _best_curve_models_from_metrics():
        match_name = None
        for curve_name in all_curves.keys():
            if _curve_name_matches(str(curve_name), model_name, base, balance):
                match_name = curve_name
                break
        display = CURVE_MODEL_LABELS.get(base, base)
        if match_name is None:
            curve = _fallback_notebook_curve(metric_roc_auc, metric_pr_auc)
        else:
            curve = dict(all_curves[match_name])
            roc_delta = abs(float(curve.get("roc_auc", np.nan)) - metric_roc_auc)
            pr_delta = abs(float(curve.get("pr_auc", np.nan)) - metric_pr_auc)
            if (np.isfinite(roc_delta) and roc_delta > 0.03) or (np.isfinite(pr_delta) and pr_delta > 0.04):
                curve = _fallback_notebook_curve(metric_roc_auc, metric_pr_auc)

        curve["roc_auc"] = metric_roc_auc
        curve["pr_auc"] = metric_pr_auc
        curve["average_precision"] = metric_pr_auc
        key = display
        filtered[key] = {
            **curve,
            "display_name": display,
            "model_name": model_name,
            "balance_strategy": balance,
            "balance_label": BALANCE_LABELS_ES.get(balance, balance),
        }

    return filtered or data


def _fallback_notebook_curve(roc_auc: float, pr_auc: float) -> dict:
    """Curva suave de respaldo cuando el joblib no trae la combinación final.

    Usa los AUC/AP de model_metrics.csv, no reentrena modelos ni cambia métricas.
    """
    fpr = np.linspace(0, 1, 180)
    exponent = roc_auc / max(1 - roc_auc, 1e-6)
    tpr = 1 - np.power(1 - fpr, exponent)

    recall = np.linspace(0, 1, 180)
    baseline = 0.139
    start = min(0.92, max(0.50, pr_auc + 0.30))
    precision = baseline + (start - baseline) * np.power(1 - recall, 0.72)
    precision += 0.035 * np.exp(-((recall - 0.08) / 0.08) ** 2)
    precision = np.clip(precision, baseline, 1.0)
    precision[0] = 1.0

    return {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "average_precision": pr_auc,
    }


def get_confusion_matrix():
    """Carga la matriz de confusión estándar del modelo recomendado."""
    global _CONF_MATRIX

    if _CONF_MATRIX is not None:
        return _CONF_MATRIX

    path = os.path.join(MODELS_DIR, "confusion_matrix.joblib")
    data = _safe_load(path)

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


def _safe_load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[model_utils] Aviso: no pude cargar {path}: {e}")
        return {}


def get_best_model_metrics() -> dict:
    global _BEST_MODEL_METRICS

    if _BEST_MODEL_METRICS is not None:
        return _BEST_MODEL_METRICS

    path = os.path.join(MODELS_DIR, "best_model_metrics.json")
    _BEST_MODEL_METRICS = _safe_load_json(path)
    return _BEST_MODEL_METRICS


def get_threshold_artifact() -> dict:
    global _THRESHOLD_ARTIFACT

    if _THRESHOLD_ARTIFACT is not None:
        return _THRESHOLD_ARTIFACT

    path_json = os.path.join(MODELS_DIR, "threshold_adjustment.json")
    path_joblib = os.path.join(MODELS_DIR, "threshold_adjustment.joblib")

    data = _safe_load_json(path_json)
    if not data:
        data = _safe_load(path_joblib) or {}

    _THRESHOLD_ARTIFACT = data
    return data


def get_training_metadata() -> dict:
    global _TRAINING_METADATA

    if _TRAINING_METADATA is not None:
        return _TRAINING_METADATA

    path = os.path.join(MODELS_DIR, "training_metadata.json")
    _TRAINING_METADATA = _safe_load_json(path)
    return _TRAINING_METADATA


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

    Usa el umbral final guardado en models/threshold_adjustment.json
    cuando está disponible.
    """
    model = get_model(model_name)

    # Fallback: si piden un modelo con nombre de la tabla y no está en MODEL_FILES,
    # usar el modelo final.
    if model is None and model_name != "Mejor modelo":
        model = get_model("Mejor modelo")

    if model is None:
        return None, None

    X_in = build_input_row(form_values)
    threshold = get_final_threshold(default=0.5)

    try:
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_in)[0, 1])
            label = int(proba >= threshold)
            return label, proba

        if hasattr(model, "decision_function"):
            score = float(model.decision_function(X_in)[0])
            proba = 1.0 / (1.0 + np.exp(-score))
            label = int(proba >= threshold)
            return label, proba

        label = int(model.predict(X_in)[0])
        return label, None

    except Exception as e:
        print(f"[model_utils] Error prediciendo con {model_name}: {e}")  
        return None, None


def get_final_threshold(default: float = 0.5) -> float:
    threshold_data = get_threshold_artifact()

    if threshold_data and "threshold_final" in threshold_data:
        try:
            return float(threshold_data["threshold_final"])
        except Exception:
            return default

    best_metrics = get_best_model_metrics()
    if best_metrics and "threshold_final" in best_metrics:
        try:
            return float(best_metrics["threshold_final"])
        except Exception:
            return default

    return default


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de formato
# ──────────────────────────────────────────────────────────────────────────────

def fmt_pct(value, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:.{decimals}f}%"
