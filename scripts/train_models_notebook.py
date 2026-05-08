"""
Entrenamiento offline de modelos para el dashboard de diabetes.

Este script replica la sección de modelado del notebook
`EDA_DIABETES_con_modelos_ML.ipynb` y guarda los artefactos necesarios para que
la pestaña de modelado del Dash pueda cargar resultados sin reentrenar al iniciar.

Salidas principales en `models/`:
    - logistic_regression_pipeline.joblib
    - knn_pipeline.joblib
    - random_forest_pipeline.joblib
    - xgboost_pipeline.joblib
    - model_metrics.csv
    - baseline_metrics.csv
    - feature_importance.csv
    - roc_pr_data.joblib
    - confusion_matrix.joblib
    - threshold_adjustment.joblib
    - classification_report_best.csv
    - feature_columns.joblib
    - best_params.json
    - training_metadata.json

Ejecutar desde la raíz del proyecto:
    python scripts/train_models.py

Nota:
    KNN se entrena con una muestra estratificada del conjunto de entrenamiento
    para reducir el costo computacional. Esta decisión conserva la proporción
    de clases y permite mantener KNN como modelo comparativo sin volver pesado
    el proceso offline del dashboard.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from typing import Any, Dict, Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Configuración general
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COL = "Diabetes_binary"
KNN_SAMPLE_SIZE = 50000

# Variables según naturaleza, tal como se documentó en el notebook.
NUMERIC_COLS = ["BMI", "MentHlth", "PhysHlth"]
ORDINAL_COLS = ["GenHlth", "Age", "Education", "Income"]

# Rutas relativas a la raíz del proyecto.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA_PATH = os.path.join(_ROOT, "data", "processed", "diabetes_clean.csv")
MODELS_DIR = os.path.join(_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Nombres canónicos usados por el dashboard.
MODEL_FILE_MAP = {
    "Regresión Logística": "logistic_regression_pipeline.joblib",
    "KNN": "knn_pipeline.joblib",
    "Random Forest": "random_forest_pipeline.joblib",
    "XGBoost": "xgboost_pipeline.joblib",
}

MODEL_NAME_FOR_DASH = {
    "Regresión Logística": "Logistic Regression",
    "KNN": "KNN",
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
}

# Utilidades

def load_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Carga la base limpia y separa predictores/objetivo."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"No se encontró el dataset en: {DATA_PATH}\n"
            "Verifica que exista data/processed/diabetes_clean.csv."
        )

    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"La columna objetivo `{TARGET_COL}` no existe en el dataset.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    return X, y, df


def make_preprocessors(binary_cols: Iterable[str]) -> Tuple[ColumnTransformer, ColumnTransformer]:
    """Crea los dos preprocesamientos usados en el notebook.

    - preprocess_scaled: para Regresión Logística y KNN, modelos sensibles a escala.
    - preprocess_tree: para Random Forest y XGBoost, modelos basados en árboles.
    """
    binary_cols = list(binary_cols)

    preprocess_scaled = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_COLS,
            ),
            (
                "ord",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ORDINAL_COLS,
            ),
            ("bin", SimpleImputer(strategy="most_frequent"), binary_cols),
        ],
        remainder="drop",
    )

    preprocess_tree = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_COLS),
            ("ord", SimpleImputer(strategy="median"), ORDINAL_COLS),
            ("bin", SimpleImputer(strategy="most_frequent"), binary_cols),
        ],
        remainder="drop",
    )

    return preprocess_scaled, preprocess_tree


def obtener_probabilidades(modelo: Any, X_eval: pd.DataFrame) -> Optional[np.ndarray]:
    """Obtiene probabilidades o puntajes para ROC-AUC y PR-AUC."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X_eval)[:, 1]
    if hasattr(modelo, "decision_function"):
        return modelo.decision_function(X_eval)
    return None


def calcular_metricas(nombre: str, modelo: Any, X_eval: pd.DataFrame, y_eval: pd.Series) -> Tuple[Dict[str, Any], np.ndarray, Optional[np.ndarray]]:
    """Calcula métricas sobre un modelo ya entrenado."""
    y_pred = modelo.predict(X_eval)
    y_score = obtener_probabilidades(modelo, X_eval)

    resultados = {
        "Modelo": nombre,
        "Accuracy": accuracy_score(y_eval, y_pred),
        "Precision": precision_score(y_eval, y_pred, zero_division=0),
        "Recall": recall_score(y_eval, y_pred, zero_division=0),
        "F1": f1_score(y_eval, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_eval, y_score) if y_score is not None else np.nan,
        "PR_AUC": average_precision_score(y_eval, y_score) if y_score is not None else np.nan,
    }

    return resultados, y_pred, y_score


def evaluar_modelo(
    nombre: str,
    modelo: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, Any], Any, np.ndarray, Optional[np.ndarray], float]:
    """Entrena un modelo y devuelve métricas, predicciones, puntajes y tiempo."""
    t0 = time.time()
    modelo.fit(X_train, y_train)
    elapsed = time.time() - t0
    resultados, y_pred, y_score = calcular_metricas(nombre, modelo, X_test, y_test)
    return resultados, modelo, y_pred, y_score, elapsed


def get_best_estimator(modelo: Any) -> Any:
    """Devuelve el pipeline final, aunque el objeto sea GridSearchCV/RandomizedSearchCV."""
    return getattr(modelo, "best_estimator_", modelo)


def get_best_params(modelo: Any) -> Optional[Dict[str, Any]]:
    return getattr(modelo, "best_params_", None)


def get_best_cv_score(modelo: Any) -> Optional[float]:
    value = getattr(modelo, "best_score_", None)
    return float(value) if value is not None else None


def to_dashboard_metrics(df_resultados: pd.DataFrame) -> pd.DataFrame:
    """Convierte nombres de columnas del notebook al formato que usa el Dash."""
    out = df_resultados.copy()
    out["model"] = out["Modelo"].map(MODEL_NAME_FOR_DASH).fillna(out["Modelo"])
    out = out.rename(
        columns={
            "Accuracy": "accuracy",
            "Precision": "precision",
            "Recall": "recall",
            "F1": "f1",
            "ROC_AUC": "roc_auc",
            "PR_AUC": "pr_auc",
        }
    )
    return out[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]


def obtener_nombres_features(preprocessor: ColumnTransformer) -> list[str]:
    """Devuelve nombres de variables después del ColumnTransformer.

    En este proyecto no se hace one-hot encoding, así que los nombres coinciden
    con las columnas originales enviadas a cada transformer.
    """
    nombres: list[str] = []
    for nombre_transformer, _, columnas in preprocessor.transformers_:
        if nombre_transformer != "remainder":
            nombres.extend(list(columnas))
    return nombres


def json_safe(obj: Any) -> Any:
    """Convierte objetos NumPy a tipos serializables para JSON."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2, ensure_ascii=False)


# Entrenamiento principal

def main() -> None:
    print("=" * 78)
    print("Entrenamiento de modelos · Diabetes BRFSS 2015")
    print("Alineado con la sección 13 del notebook EDA_DIABETES_con_modelos_ML.ipynb")
    print("=" * 78)

    X, y, df = load_data()
    print(f"\nDimensiones del dataset: {df.shape}")
    print("Distribución de la variable objetivo (%):")
    print((y.value_counts(normalize=True).sort_index() * 100).round(2).to_string())

    binary_cols = [col for col in X.columns if col not in NUMERIC_COLS + ORDINAL_COLS]
    print(f"\nVariables numéricas: {NUMERIC_COLS}")
    print(f"Variables ordinales: {ORDINAL_COLS}")
    print(f"Variables binarias ({len(binary_cols)}): {binary_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    conteo = y.value_counts().sort_index()
    split_info = {
        "dataset_shape": list(df.shape),
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
        "target_distribution_total": y.value_counts(normalize=True).sort_index().to_dict(),
        "target_distribution_train": y_train.value_counts(normalize=True).sort_index().to_dict(),
        "target_distribution_test": y_test.value_counts(normalize=True).sort_index().to_dict(),
        "class_balance_ratio_neg_pos": float(conteo.loc[0] / conteo.loc[1]),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }

    print(f"\nTamaño de entrenamiento: {X_train.shape}")
    print(f"Tamaño de prueba: {X_test.shape}")
    print(f"Razón de desbalance (0/1): {split_info['class_balance_ratio_neg_pos']:.2f}:1")

    preprocess_scaled, preprocess_tree = make_preprocessors(binary_cols)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Guardar el orden de columnas para el simulador del dashboard.
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "feature_columns.joblib"))

    results_all: list[Dict[str, Any]] = []
    results_models: list[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}
    predictions: Dict[str, np.ndarray] = {}
    scores: Dict[str, Optional[np.ndarray]] = {}
    best_params: Dict[str, Any] = {}
    cv_scores: Dict[str, Any] = {}
    fit_times: Dict[str, float] = {}

    # ── 0. Dummy Classifier ───────────────────────────────────────────────────
    print("\n[0/4] Modelo base: Dummy Classifier")
    dummy_model = Pipeline(
        steps=[
            ("preprocess", preprocess_tree),
            ("model", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
        ]
    )
    res_dummy, dummy_model, y_pred_dummy, y_score_dummy, elapsed = evaluar_modelo(
        "Dummy Classifier", dummy_model, X_train, y_train, X_test, y_test
    )
    results_all.append(res_dummy)
    fit_times["Dummy Classifier"] = elapsed
    print(f"   F1 test: {res_dummy['F1']:.4f} | Accuracy: {res_dummy['Accuracy']:.4f} | t={elapsed:.1f}s")

    # ── 1. Regresión Logística ───────────────────────────────────────────────
    print("\n[1/4] Regresión Logística con GridSearchCV")
    pipe_logreg = Pipeline(
        steps=[
            ("preprocess", preprocess_scaled),
            ("model", LogisticRegression(max_iter=10000, random_state=RANDOM_STATE)),
        ]
    )
    param_logreg = {
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "model__penalty": ["l2"],
        "model__class_weight": [None, "balanced"],
    }
    grid_logreg = GridSearchCV(
        estimator=pipe_logreg,
        param_grid=param_logreg,
        scoring="f1",
        cv=skf,
        n_jobs=-1,
        verbose=1,
    )
    res_logreg, best_logreg, y_pred_logreg, y_score_logreg, elapsed = evaluar_modelo(
        "Regresión Logística", grid_logreg, X_train, y_train, X_test, y_test
    )
    results_all.append(res_logreg)
    results_models.append(res_logreg)
    trained_models["Regresión Logística"] = best_logreg
    predictions["Regresión Logística"] = y_pred_logreg
    scores["Regresión Logística"] = y_score_logreg
    best_params["Regresión Logística"] = get_best_params(best_logreg)
    cv_scores["Regresión Logística"] = get_best_cv_score(best_logreg)
    fit_times["Regresión Logística"] = elapsed
    print(f"   Mejores hiperparámetros: {best_params['Regresión Logística']}")
    print(f"   F1 CV: {cv_scores['Regresión Logística']:.4f} | F1 test: {res_logreg['F1']:.4f} | t={elapsed:.1f}s")

    # ── 2. KNN ───────────────────────────────────────────────────────────────
    print("\n[2/4] KNN con GridSearchCV y muestra estratificada")

    # KNN es costoso en datasets grandes porque calcula distancias durante la
    # validación y predicción. Por eso se usa una muestra estratificada de
    # entrenamiento, conservando la proporción original de clases.
    n_muestra_knn = min(KNN_SAMPLE_SIZE, len(X_train))
    X_train_knn, _, y_train_knn, _ = train_test_split(
        X_train,
        y_train,
        train_size=n_muestra_knn,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    print(f"   Tamaño muestra KNN: {X_train_knn.shape}")
    print("   Distribución de clases en muestra KNN (%):")
    print((y_train_knn.value_counts(normalize=True).sort_index() * 100).round(2).to_string())

    pipe_knn = Pipeline(
        steps=[
            ("preprocess", preprocess_scaled),
            ("model", KNeighborsClassifier(n_jobs=-1)),
        ]
    )
    param_knn = {
        "model__n_neighbors": [5, 11, 21, 51, 101],
        "model__weights": ["uniform", "distance"],
        "model__metric": ["minkowski", "manhattan"],
    }
    grid_knn = GridSearchCV(
        estimator=pipe_knn,
        param_grid=param_knn,
        scoring="f1",
        cv=skf,
        n_jobs=-1,
        verbose=1,
    )
    res_knn, best_knn, y_pred_knn, y_score_knn, elapsed = evaluar_modelo(
        "KNN", grid_knn, X_train_knn, y_train_knn, X_test, y_test
    )
    results_all.append(res_knn)
    results_models.append(res_knn)
    trained_models["KNN"] = best_knn
    predictions["KNN"] = y_pred_knn
    scores["KNN"] = y_score_knn
    best_params["KNN"] = get_best_params(best_knn)
    cv_scores["KNN"] = get_best_cv_score(best_knn)
    fit_times["KNN"] = elapsed
    print(f"   Mejores hiperparámetros: {best_params['KNN']}")
    print(f"   F1 CV: {cv_scores['KNN']:.4f} | F1 test: {res_knn['F1']:.4f} | t={elapsed:.1f}s")

    # ── 3. Random Forest ─────────────────────────────────────────────────────
    print("\n[3/4] Random Forest con RandomizedSearchCV")
    pipe_rf = Pipeline(
        steps=[
            ("preprocess", preprocess_tree),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    param_rf = {
        "model__n_estimators": [150, 250, 350],
        "model__max_depth": [None, 8, 12, 16],
        "model__min_samples_split": [2, 10, 30],
        "model__min_samples_leaf": [1, 5, 10],
        "model__max_features": ["sqrt", "log2"],
        "model__class_weight": ["balanced", "balanced_subsample"],
    }
    random_rf = RandomizedSearchCV(
        estimator=pipe_rf,
        param_distributions=param_rf,
        n_iter=15,
        scoring="f1",
        cv=skf,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    res_rf, best_rf, y_pred_rf, y_score_rf, elapsed = evaluar_modelo(
        "Random Forest", random_rf, X_train, y_train, X_test, y_test
    )
    results_all.append(res_rf)
    results_models.append(res_rf)
    trained_models["Random Forest"] = best_rf
    predictions["Random Forest"] = y_pred_rf
    scores["Random Forest"] = y_score_rf
    best_params["Random Forest"] = get_best_params(best_rf)
    cv_scores["Random Forest"] = get_best_cv_score(best_rf)
    fit_times["Random Forest"] = elapsed
    print(f"   Mejores hiperparámetros: {best_params['Random Forest']}")
    print(f"   F1 CV: {cv_scores['Random Forest']:.4f} | F1 test: {res_rf['F1']:.4f} | t={elapsed:.1f}s")

    # ── 4. XGBoost ───────────────────────────────────────────────────────────
    print("\n[4/4] XGBoost con RandomizedSearchCV y scale_pos_weight")
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"   scale_pos_weight calculado: {scale_pos_weight:.4f}")

    pipe_xgb = Pipeline(
        steps=[
            ("preprocess", preprocess_tree),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    tree_method="hist",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    param_xgb = {
        "model__n_estimators": [150, 250, 300, 350],
        "model__max_depth": [3, 4, 5],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__min_child_weight": [1, 5, 10],
        "model__scale_pos_weight": [1, scale_pos_weight],
    }
    random_xgb = RandomizedSearchCV(
        estimator=pipe_xgb,
        param_distributions=param_xgb,
        n_iter=20,
        scoring="f1",
        cv=skf,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    res_xgb, best_xgb, y_pred_xgb, y_score_xgb, elapsed = evaluar_modelo(
        "XGBoost", random_xgb, X_train, y_train, X_test, y_test
    )
    results_all.append(res_xgb)
    results_models.append(res_xgb)
    trained_models["XGBoost"] = best_xgb
    predictions["XGBoost"] = y_pred_xgb
    scores["XGBoost"] = y_score_xgb
    best_params["XGBoost"] = get_best_params(best_xgb)
    cv_scores["XGBoost"] = get_best_cv_score(best_xgb)
    fit_times["XGBoost"] = elapsed
    print(f"   Mejores hiperparámetros: {best_params['XGBoost']}")
    print(f"   F1 CV: {cv_scores['XGBoost']:.4f} | F1 test: {res_xgb['F1']:.4f} | t={elapsed:.1f}s")

   
    # Comparación general y persistencia de resultados
    print("\nGuardando artefactos en models/ ...")

    df_resultados = (
        pd.DataFrame(results_models)
        .sort_values("F1", ascending=False)
        .reset_index(drop=True)
    )
    df_resultados_all = (
        pd.DataFrame(results_all)
        .sort_values("F1", ascending=False)
        .reset_index(drop=True)
    )

    metrics_dash = to_dashboard_metrics(df_resultados)
    metrics_dash.to_csv(os.path.join(MODELS_DIR, "model_metrics.csv"), index=False)

    baseline_dash = to_dashboard_metrics(pd.DataFrame([res_dummy]))
    baseline_dash.to_csv(os.path.join(MODELS_DIR, "baseline_metrics.csv"), index=False)

    # También guardamos una versión más cercana al notebook para auditoría.
    df_resultados.to_csv(os.path.join(MODELS_DIR, "model_metrics_notebook_format.csv"), index=False)
    df_resultados_all.to_csv(os.path.join(MODELS_DIR, "model_metrics_with_dummy.csv"), index=False)

    print("\nTabla final de modelos:")
    print(df_resultados.round(4).to_string(index=False))

    # Guardar pipelines optimizados. Se guarda el best_estimator_, no el objeto SearchCV,
    # para que el dashboard cargue pipelines más livianos y listos para predecir.
    for nombre, modelo in trained_models.items():
        final_pipeline = get_best_estimator(modelo)
        path = os.path.join(MODELS_DIR, MODEL_FILE_MAP[nombre])
        joblib.dump(final_pipeline, path)
        print(f"  · {nombre} → {path}")

    # Datos para curvas ROC y Precision-Recall.
    roc_pr_data: Dict[str, Dict[str, Any]] = {}
    for nombre, y_score in scores.items():
        if y_score is None:
            continue
        canonical = MODEL_NAME_FOR_DASH.get(nombre, nombre)
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_score)
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_score)
        roc_pr_data[canonical] = {
            "fpr": fpr,
            "tpr": tpr,
            "roc_thresholds": roc_thresholds,
            "precision": precision,
            "recall": recall,
            "pr_thresholds": pr_thresholds,
            "roc_auc": roc_auc_score(y_test, y_score),
            "pr_auc": average_precision_score(y_test, y_score),
        }
    joblib.dump(roc_pr_data, os.path.join(MODELS_DIR, "roc_pr_data.joblib"))

    # Selección del mejor modelo según F1-score.
    mejor_modelo_nombre = str(df_resultados.loc[0, "Modelo"])
    mejor_modelo_nombre_dash = MODEL_NAME_FOR_DASH.get(mejor_modelo_nombre, mejor_modelo_nombre)
    mejor_modelo_search = trained_models[mejor_modelo_nombre]
    mejor_modelo_pipeline = get_best_estimator(mejor_modelo_search)
    y_pred_mejor = predictions[mejor_modelo_nombre]
    y_score_mejor = scores[mejor_modelo_nombre]

    print(f"\nMejor modelo según F1-score: {mejor_modelo_nombre}")

    # Reporte de clasificación del mejor modelo.
    report_dict = classification_report(
        y_test,
        y_pred_mejor,
        target_names=["Sin diabetes", "Prediabetes / Diabetes"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(MODELS_DIR, "classification_report_best.csv"), index=True)

    # Matriz de confusión con umbral estándar 0.5.
    cm = confusion_matrix(y_test, y_pred_mejor)
    joblib.dump(
        {
            "best_model": mejor_modelo_nombre_dash,
            "best_model_notebook": mejor_modelo_nombre,
            "matrix": cm,
            "labels": ["Sin diabetes", "Prediabetes / Diabetes"],
            "threshold": 0.5,
        },
        os.path.join(MODELS_DIR, "confusion_matrix.joblib"),
    )

    # Ajuste opcional de umbral para maximizar F1 del mejor modelo.
    threshold_artifact: Dict[str, Any] = {
        "best_model": mejor_modelo_nombre_dash,
        "best_model_notebook": mejor_modelo_nombre,
        "available": False,
    }
    if y_score_mejor is not None:
        precision, recall, thresholds = precision_recall_curve(y_test, y_score_mejor)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        mejor_idx = int(np.argmax(f1_scores))
        mejor_threshold = float(thresholds[mejor_idx])
        y_pred_umbral = (y_score_mejor >= mejor_threshold).astype(int)
        cm_umbral = confusion_matrix(y_test, y_pred_umbral)
        report_umbral = classification_report(
            y_test,
            y_pred_umbral,
            target_names=["Sin diabetes", "Prediabetes / Diabetes"],
            output_dict=True,
            zero_division=0,
        )
        threshold_artifact = {
            "best_model": mejor_modelo_nombre_dash,
            "best_model_notebook": mejor_modelo_nombre,
            "available": True,
            "threshold": mejor_threshold,
            "f1": float(f1_scores[mejor_idx]),
            "precision": float(precision[mejor_idx]),
            "recall": float(recall[mejor_idx]),
            "confusion_matrix": cm_umbral,
            "classification_report": report_umbral,
            "labels": ["Sin diabetes", "Prediabetes / Diabetes"],
        }
        print(
            "Umbral ajustado: "
            f"threshold={mejor_threshold:.4f} | F1={f1_scores[mejor_idx]:.4f} | "
            f"precision={precision[mejor_idx]:.4f} | recall={recall[mejor_idx]:.4f}"
        )
    joblib.dump(threshold_artifact, os.path.join(MODELS_DIR, "threshold_adjustment.joblib"))

    # Importancia de variables del mejor modelo basado en árboles disponible.
    modelo_importancia = None
    nombre_importancia = None
    if mejor_modelo_nombre in ["Random Forest", "XGBoost"]:
        modelo_importancia = mejor_modelo_pipeline
        nombre_importancia = mejor_modelo_nombre_dash
    elif "XGBoost" in trained_models:
        modelo_importancia = get_best_estimator(trained_models["XGBoost"])
        nombre_importancia = "XGBoost"
    elif "Random Forest" in trained_models:
        modelo_importancia = get_best_estimator(trained_models["Random Forest"])
        nombre_importancia = "Random Forest"

    if modelo_importancia is not None:
        preprocessor_fit = modelo_importancia.named_steps["preprocess"]
        estimator_fit = modelo_importancia.named_steps["model"]
        if hasattr(estimator_fit, "feature_importances_"):
            feature_names = obtener_nombres_features(preprocessor_fit)
            importances = estimator_fit.feature_importances_
            importancia_df = (
                pd.DataFrame({"feature": feature_names, "importance": importances})
                .sort_values(by="importance", ascending=False)
                .reset_index(drop=True)
            )
            importancia_df["model"] = nombre_importancia
            importancia_df.to_csv(os.path.join(MODELS_DIR, "feature_importance.csv"), index=False)
            # versión con nombres del notebook
            importancia_df.rename(columns={"feature": "Variable", "importance": "Importancia"}).to_csv(
                os.path.join(MODELS_DIR, "feature_importance_notebook_format.csv"), index=False
            )
            print(f"  · feature_importance.csv ({nombre_importancia})")

    # Guardar mejores hiperparámetros y metadatos.
    save_json(
        {
            "best_params": best_params,
            "best_cv_scores_f1": cv_scores,
            "fit_times_seconds": fit_times,
            "best_model_by_f1": mejor_modelo_nombre,
            "best_model_by_f1_dashboard_name": mejor_modelo_nombre_dash,
            "scale_pos_weight_xgb": float(scale_pos_weight),
        },
        os.path.join(MODELS_DIR, "best_params.json"),
    )

    save_json(
        {
            "split_info": split_info,
            "numeric_cols": NUMERIC_COLS,
            "ordinal_cols": ORDINAL_COLS,
            "binary_cols": binary_cols,
            "knn_sample_size_configured": KNN_SAMPLE_SIZE,
            "knn_sample_size_used": int(min(KNN_SAMPLE_SIZE, len(X_train))),
            "output_files": sorted(os.listdir(MODELS_DIR)),
        },
        os.path.join(MODELS_DIR, "training_metadata.json"),
    )

    print("\nArchivos principales generados:")
    for fname in [
        "model_metrics.csv",
        "baseline_metrics.csv",
        "logistic_regression_pipeline.joblib",
        "knn_pipeline.joblib",
        "random_forest_pipeline.joblib",
        "xgboost_pipeline.joblib",
        "roc_pr_data.joblib",
        "confusion_matrix.joblib",
        "threshold_adjustment.joblib",
        "feature_importance.csv",
        "feature_columns.joblib",
        "best_params.json",
        "training_metadata.json",
    ]:
        print(f"  - models/{fname}")

    print("\n Entrenamiento completado.")


if __name__ == "__main__":
    main()
