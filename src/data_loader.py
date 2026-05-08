import os
import sys
import pandas as pd
import numpy as np

# ── Ruta robusta al CSV ───────────────────────────────────────────────────────
# Sube dos niveles desde este archivo: src/ -> raíz del proyecto
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
DATA_PATH = os.path.join(_ROOT_DIR, "data", "processed", "diabetes_clean.csv")

# ── Órdenes y mappings ────────────────────────────────────────────────────────
AGE_ORDER = [
    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
    "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+",
]
AGE_MAP = {
    1: "18-24", 2: "25-29", 3: "30-34",  4: "35-39",  5: "40-44",
    6: "45-49", 7: "50-54", 8: "55-59",  9: "60-64", 10: "65-69",
    11: "70-74", 12: "75-79", 13: "80+",
}

BINARY_MAPPINGS = {
    "HighBP":               {0: "Sin hipertension",    1: "Con hipertension"},
    "HighChol":             {0: "Sin colesterol alto", 1: "Con colesterol alto"},
    "CholCheck":            {0: "Sin chequeo",         1: "Con chequeo"},
    "Smoker":               {0: "No fumador",          1: "Fumador"},
    "Stroke":               {0: "Sin ACV",             1: "Con ACV"},
    "HeartDiseaseorAttack": {0: "Sin enf. cardiaca",   1: "Con enf. cardiaca"},
    "PhysActivity":         {0: "Sin actividad",       1: "Con actividad"},
    "Fruits":               {0: "No consume frutas",   1: "Consume frutas"},
    "Veggies":              {0: "No consume verduras",  1: "Consume verduras"},
    "HvyAlcoholConsump":    {0: "Sin alcohol alto",    1: "Consumo alto alcohol"},
    "AnyHealthcare":        {0: "Sin cobertura",       1: "Con cobertura"},
    "NoDocbcCost":          {0: "Sin barrera",         1: "Con barrera economica"},
    "DiffWalk":             {0: "Sin dific. caminar",  1: "Con dific. caminar"},
    "Sex":                  {0: "Mujer",               1: "Hombre"},
}

GENHLTH_MAP   = {1: "Excelente", 2: "Muy buena", 3: "Buena", 4: "Regular", 5: "Mala"}
GENHLTH_ORDER = ["Excelente", "Muy buena", "Buena", "Regular", "Mala"]

EDUCATION_MAP = {
    1: "Nunca asistio",     2: "Primaria incompleta",
    3: "Sec. incompleta",   4: "Sec. completa/GED",
    5: "Univ. incompleta",  6: "Univ. completa",
}
EDUCATION_ORDER = list(EDUCATION_MAP.values())

INCOME_MAP = {
    1: "< $10k",    2: "$10k-15k", 3: "$15k-20k", 4: "$20k-25k",
    5: "$25k-35k",  6: "$35k-50k", 7: "$50k-75k",  8: "$75k+",
}
INCOME_ORDER = list(INCOME_MAP.values())

FACTOR_OPTIONS = [
    {"label": "Hipertension",          "value": "HighBP"},
    {"label": "Colesterol alto",       "value": "HighChol"},
    {"label": "Fumador",               "value": "Smoker"},
    {"label": "Actividad fisica",      "value": "PhysActivity"},
    {"label": "Enf. cardiaca",         "value": "HeartDiseaseorAttack"},
    {"label": "Dificultad al caminar", "value": "DiffWalk"},
    {"label": "Consumo frutas",        "value": "Fruits"},
    {"label": "Consumo verduras",      "value": "Veggies"},
    {"label": "Alcohol elevado",       "value": "HvyAlcoholConsump"},
    {"label": "Cobertura medica",      "value": "AnyHealthcare"},
    {"label": "Barrera economica",     "value": "NoDocbcCost"},
    {"label": "Salud general",         "value": "GenHlth"},
    {"label": "Grupo de edad",         "value": "Age"},
    {"label": "Educacion",             "value": "Education"},
    {"label": "Ingreso",               "value": "Income"},
]


# ── Carga y transformación ────────────────────────────────────────────────────

def load_raw_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\n\n[ERROR] CSV no encontrado en:\n  {DATA_PATH}\n\n"
            "Coloca el archivo 'diabetes_clean.csv' en la carpeta:\n"
            f"  {os.path.join(_ROOT_DIR, 'data', 'processed')}\n"
        )
    return pd.read_csv(DATA_PATH)


def build_viz_df(df):
    dv = df.copy()

    dv["Diabetes_binary"] = (
        dv["Diabetes_binary"]
        .map({0: "Sin diabetes", 1: "Prediabetes / Diabetes"})
        .astype("category")
    )

    for col, mapping in BINARY_MAPPINGS.items():
        if col in dv.columns:
            dv[col] = dv[col].map(mapping).astype("category")

    if "GenHlth" in dv.columns:
        dv["GenHlth"] = (
            dv["GenHlth"].map(GENHLTH_MAP).astype("category")
            .cat.reorder_categories(GENHLTH_ORDER, ordered=True)
        )

    if "Age" in dv.columns:
        dv["Age"] = (
            dv["Age"].replace(99, np.nan).map(AGE_MAP).astype("category")
            .cat.reorder_categories(AGE_ORDER, ordered=True)
        )

    if "Education" in dv.columns:
        dv["Education"] = (
            dv["Education"].map(EDUCATION_MAP).astype("category")
            .cat.reorder_categories(EDUCATION_ORDER, ordered=True)
        )

    if "Income" in dv.columns:
        dv["Income"] = (
            dv["Income"].replace({77: np.nan, 99: np.nan}).map(INCOME_MAP)
            .astype("category")
            .cat.reorder_categories(INCOME_ORDER, ordered=True)
        )

    return dv


def get_data():
    raw = load_raw_data()
    viz = build_viz_df(raw)
    return raw, viz
