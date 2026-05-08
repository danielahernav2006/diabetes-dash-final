import pandas as pd
import numpy as np
from src.data_loader import (
    AGE_ORDER, GENHLTH_ORDER, EDUCATION_ORDER, INCOME_ORDER,
    BINARY_MAPPINGS,
)

# ─────────────────────────────────────────────
# FILTERING
# ─────────────────────────────────────────────

def apply_filters(df_viz, sex_filter, age_filter):
    dff = df_viz.copy()
    if sex_filter and sex_filter != "Todos":
        if "Sex" in dff.columns:
            dff = dff[dff["Sex"] == sex_filter]
    if age_filter and len(age_filter) > 0:
        if "Age" in dff.columns:
            dff = dff[dff["Age"].isin(age_filter)]
    return dff


# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────

def compute_kpis(df_viz):
    total = len(df_viz)
    if total == 0:
        return {
            "total": "0", "total_sub": "sin registros",
            "prevalencia": "—", "prevalencia_sub": "sin datos",
            "bmi_avg": "—", "bmi_sub": "—",
            "age_label": "—", "age_sub": "—",
        }

    dm = df_viz["Diabetes_binary"] == "Prediabetes / Diabetes"
    prevalence = round(dm.mean() * 100, 1)
    bmi_avg = round(df_viz["BMI"].mean(), 1) if "BMI" in df_viz.columns else 0
    bmi_std = round(df_viz["BMI"].std(), 1) if "BMI" in df_viz.columns else 0

    # Grupo de edad más frecuente
    age_label = "N/A"
    if "Age" in df_viz.columns:
        age_counts = df_viz["Age"].value_counts()
        if len(age_counts) > 0:
            age_label = str(age_counts.index[0])

    ratio_str = f"1 de cada {round(100/prevalence)}" if prevalence > 0 else "—"
    bmi_cat = "obesidad" if bmi_avg >= 30 else "sobrepeso" if bmi_avg >= 25 else "normal"

    return {
        "total":           f"{total:,}",
        "total_sub":       "individuos en la selección actual",
        "prevalencia":     f"{prevalence}%",
        "prevalencia_sub": f"{ratio_str} individuos presenta diabetes",
        "bmi_avg":         f"{bmi_avg} ± {bmi_std}",
        "bmi_sub":         f"Promedio ± desviación estándar · {bmi_cat}",
        "age_label":       age_label,
        "age_sub":         "grupo etario más representado",
    }


# ─────────────────────────────────────────────
# FACTOR RANKING
# ─────────────────────────────────────────────

FACTOR_RANK_COLS = {
    "HighBP":               "Hipertensión",
    "HighChol":             "Colesterol alto",
    "HeartDiseaseorAttack": "Enf. cardíaca",
    "DiffWalk":             "Dificultad al caminar",
    "Smoker":               "Tabaquismo",
    "PhysActivity":         "Actividad física",
    "Fruits":               "Consumo de frutas",
    "Veggies":              "Consumo de verduras",
    "HvyAlcoholConsump":    "Consumo alto alcohol",
    "AnyHealthcare":        "Cobertura médica",
    "NoDocbcCost":          "Barrera económica",
    "Stroke":               "ACV previo",
    "CholCheck":            "Chequeo colesterol",
}


def compute_factor_ranking(df_viz):
    if df_viz.empty:
        return pd.DataFrame(columns=[
            "Factor", "Col", "Prevalencia con DM", "Prevalencia sin DM", "Diferencia (pp)"
        ])

    dm = df_viz["Diabetes_binary"] == "Prediabetes / Diabetes"
    rows = []
    for col, label in FACTOR_RANK_COLS.items():
        if col not in df_viz.columns:
            continue
        try:
            pos_val = list(BINARY_MAPPINGS[col].values())[1]
            grp_dm = df_viz[dm][col]
            grp_no = df_viz[~dm][col]
            if len(grp_dm) == 0 or len(grp_no) == 0:
                continue
            r_diab = (grp_dm == pos_val).mean() * 100
            r_no   = (grp_no == pos_val).mean() * 100
            rows.append({
                "Factor":             label,
                "Col":                col,
                "Prevalencia con DM": round(r_diab, 1),
                "Prevalencia sin DM": round(r_no, 1),
                "Diferencia (pp)":    round(r_diab - r_no, 1),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=[
            "Factor", "Col", "Prevalencia con DM", "Prevalencia sin DM", "Diferencia (pp)"
        ])

    return (
        pd.DataFrame(rows)
        .sort_values("Diferencia (pp)", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────

def _get_factor_order(factor):
    return {
        "GenHlth":   GENHLTH_ORDER,
        "Age":       AGE_ORDER,
        "Education": EDUCATION_ORDER,
        "Income":    INCOME_ORDER,
    }.get(factor)


def compute_summary_table(df_viz, factor):
    if not factor or factor not in df_viz.columns or df_viz.empty:
        return pd.DataFrame()

    try:
        tabla_abs = pd.crosstab(df_viz[factor], df_viz["Diabetes_binary"])
    except Exception:
        return pd.DataFrame()

    order = _get_factor_order(factor)
    if order:
        existing = [o for o in order if o in tabla_abs.index]
        if existing:
            tabla_abs = tabla_abs.reindex(existing)

    total = tabla_abs.sum(axis=1)
    tabla_pct = tabla_abs.div(total, axis=0) * 100

    result = pd.DataFrame({"Categoría": tabla_abs.index.astype(str)})
    for col in ["Sin diabetes", "Prediabetes / Diabetes"]:
        if col in tabla_abs.columns:
            result[f"{col} (N)"] = tabla_abs[col].values
            result[f"{col} (%)"] = tabla_pct[col].round(1).values
    result["Total (N)"] = total.values
    return result.reset_index(drop=True)


# ─────────────────────────────────────────────
# EXECUTIVE SUMMARY CARDS
# ─────────────────────────────────────────────

def compute_executive_summary(df_viz):
    if df_viz.empty:
        return []

    dm = df_viz["Diabetes_binary"] == "Prediabetes / Diabetes"
    prevalence = dm.mean() * 100
    ratio = round(100 / prevalence) if prevalence > 0 else 0

    bmi_dm  = df_viz[dm]["BMI"].mean() if "BMI" in df_viz.columns else 0
    bmi_no  = df_viz[~dm]["BMI"].mean() if "BMI" in df_viz.columns else 0
    bmi_gap = round(bmi_dm - bmi_no, 1)

    ranking    = compute_factor_ranking(df_viz)
    top_factor = ranking.iloc[0] if len(ranking) > 0 else None

    phys_diff        = 0
    phys_dm_inactive = 0
    if "PhysActivity" in df_viz.columns:
        try:
            no_phys = list(BINARY_MAPPINGS["PhysActivity"].values())[0]
            phys_dm_inactive = round(
                (df_viz[dm]["PhysActivity"] == no_phys).mean() * 100, 1
            )
            phys_no_inactive = round(
                (df_viz[~dm]["PhysActivity"] == no_phys).mean() * 100, 1
            )
            phys_diff = round(phys_dm_inactive - phys_no_inactive, 1)
        except Exception:
            pass

    cards = []

    cards.append({
        "tipo":   "critico",
        "titulo": "Prevalencia poblacional",
        "valor":  f"{round(prevalence, 1)}%",
        "texto":  (
            f"1 de cada {ratio} individuos presenta diabetes o prediabetes. "
            f"Equivale a {int(dm.sum()):,} casos sobre {len(df_viz):,} registros analizados."
        ),
    })

    if top_factor is not None:
        cards.append({
            "tipo":   "alerta",
            "titulo": "Factor de riesgo principal",
            "valor":  top_factor["Factor"],
            "texto":  (
                f"Prevalencia con {top_factor['Factor'].lower()}: "
                f"{top_factor['Prevalencia con DM']}% vs. "
                f"{top_factor['Prevalencia sin DM']}% sin dicho factor. "
                f"Diferencia de {abs(top_factor['Diferencia (pp)'])} puntos porcentuales."
            ),
        })

    cards.append({
        "tipo":   "alerta",
        "titulo": "IMC como indicador metabólico",
        "valor":  f"+{bmi_gap} pts IMC",
        "texto":  (
            f"IMC promedio con diabetes: {round(bmi_dm, 1)} "
            f"({bmi_gap} puntos mayor que el grupo sin diabetes: {round(bmi_no, 1)}). "
            f"IMC >= 30 clasifica como obesidad."
        ),
    })

    if phys_diff > 0:
        cards.append({
            "tipo":   "positivo",
            "titulo": "Sedentarismo y diabetes",
            "valor":  f"+{phys_diff} pp",
            "texto":  (
                f"{phys_dm_inactive}% de los individuos con diabetes reporta inactividad física, "
                f"{phys_diff} pp más que en el grupo sin diabetes. "
                f"La actividad física actúa como factor protector."
            ),
        })

    return cards


# ─────────────────────────────────────────────
# INSIGHTS TEXT
# ─────────────────────────────────────────────

def compute_insights(df_viz):
    if df_viz.empty:
        return "Sin datos para los filtros seleccionados."

    dm         = df_viz["Diabetes_binary"] == "Prediabetes / Diabetes"
    total      = len(df_viz)
    prevalence = round(dm.mean() * 100, 1)
    bmi_dm     = round(df_viz[dm]["BMI"].mean(), 1) if "BMI" in df_viz.columns else "—"
    bmi_no     = round(df_viz[~dm]["BMI"].mean(), 1) if "BMI" in df_viz.columns else "—"
    bmi_gap    = round(bmi_dm - bmi_no, 1) if isinstance(bmi_dm, float) else "—"

    ranking = compute_factor_ranking(df_viz)
    top     = ranking.iloc[0] if len(ranking) > 0 else None

    lines = [
        f"**Prevalencia:** {prevalence}% de los {total:,} individuos "
        f"seleccionados presentan diabetes o prediabetes.",
    ]

    if top is not None:
        lines.append(
            f"**Factor de mayor diferenciación:** {top['Factor']} — "
            f"prevalencia de {top['Prevalencia con DM']}% vs. "
            f"{top['Prevalencia sin DM']}% "
            f"({abs(top['Diferencia (pp)'])} pp de diferencia)."
        )

    if isinstance(bmi_dm, float):
        lines.append(
            f"**IMC:** El grupo con diabetes registra un IMC promedio de {bmi_dm}, "
            f"{bmi_gap} puntos superior al grupo sin diabetes ({bmi_no}), "
            f"situándose en rango de "
            f"{'obesidad' if bmi_dm >= 30 else 'sobrepeso'}."
        )

    if len(ranking) >= 3:
        top3 = ", ".join(ranking.head(3)["Factor"].tolist())
        lines.append(
            f"**Factores más diferenciadores:** {top3}. "
            f"Concentrar intervenciones en estos perfiles maximiza el impacto preventivo."
        )

    return "\n\n".join(lines)
