"""
Figuras Plotly para la pestaña de modelado predictivo.

Mantiene la paleta y la línea visual de `src/utils/figures.py` para que los
gráficos del módulo de ML se sientan parte del mismo dashboard.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.model_utils import (
    FEATURE_LABELS_ES,
    get_feature_importance,
    get_metrics_df,
    get_roc_pr_data,
    get_confusion_matrix
)

# ── Paleta consistente con figures.py ────────────────────────────────────────
C_BLUE   = "#17345F"
C_RED    = "#B83A32"
C_BLUE_L = "#4A7FC1"
C_GRAY   = "#64748B"
C_GRID   = "#E5EAF2"
C_BG     = "#FFFFFF"

# Color por modelo (estable a través de todos los gráficos)
MODEL_COLORS = {
    "Random Forest":        C_BLUE,
    "XGBoost":              C_RED,
    "Logistic Regression":  C_BLUE_L,
    "Regresión Logística":  C_BLUE_L,
    "KNN":                  "#C98A16",  # ámbar
    "XGBoost + class_weight": "#B83A32",
    "Random Forest + class_weight": "#1B3A6B",
    "Regresión Logística + class_weight": "#4A7FC1",
    "KNN + adasyn": "#C98A16",
}


def _empty_fig(message: str = "Sin datos disponibles") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=13, color=C_GRAY),
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=C_BG, plot_bgcolor=C_BG,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def _base_layout(title: str = "", subtitle: str = "", height: Optional[int] = None) -> dict:
    full_title = f"<b>{title}</b>" if title else ""
    if subtitle:
        full_title += f"<br><span style='font-size:11px;color:{C_GRAY}'>{subtitle}</span>"
    layout = dict(
        template="plotly_white",
        font=dict(family="Inter, Arial, sans-serif", size=12, color="#334155"),
        paper_bgcolor=C_BG, plot_bgcolor=C_BG,
        margin=dict(l=18, r=18, t=70 if title else 30, b=18),
        title=dict(text=full_title, x=0.02, xanchor="left",
                   font=dict(size=14, color="#0B1628")) if title else None,
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)", bordercolor=C_GRID, borderwidth=1,
            font=dict(size=11, color="#334155"),
        ),
        hoverlabel=dict(bgcolor="#0B1628", bordercolor="#0B1628",
                        font=dict(color="#FFFFFF", size=12)),
    )
    if height:
        layout["height"] = height
    return layout


def _grid_axes(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(gridcolor=C_GRID, zerolinecolor=C_GRID,
                     linecolor="#CBD5E1", tickfont=dict(color="#475569"))
    fig.update_yaxes(gridcolor=C_GRID, zerolinecolor=C_GRID,
                     linecolor="#CBD5E1", tickfont=dict(color="#475569"))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 1) Matriz de confusión (mejor modelo)
# ──────────────────────────────────────────────────────────────────────────────

def fig_confusion_matrix():
    """Construye la matriz de confusión estándar del modelo recomendado."""

    import numpy as np
    import plotly.graph_objects as go

    cm_data = get_confusion_matrix()

    if not cm_data:
        return _empty_fig("Matriz de confusión no disponible")

    # ------------------------------------------------------------------
    # Compatibilidad con varios formatos de guardado
    # ------------------------------------------------------------------
    if isinstance(cm_data, dict):
        if "confusion_matrix_default" in cm_data:
            matrix = cm_data["confusion_matrix_default"]
            threshold = cm_data.get("default_threshold", 0.5)

        elif "matrix" in cm_data:
            matrix = cm_data["matrix"]
            threshold = cm_data.get("threshold", 0.5)

        elif "confusion_matrix" in cm_data:
            matrix = cm_data["confusion_matrix"]
            threshold = cm_data.get("threshold", 0.5)

        else:
            return _empty_fig("Matriz de confusión no disponible")

        labels = cm_data.get(
            "labels",
            ["Sin diabetes", "Prediabetes / Diabetes"]
        )

    else:
        matrix = cm_data
        labels = ["Sin diabetes", "Prediabetes / Diabetes"]
        threshold = 0.5

    matrix = np.array(matrix)

    if matrix.shape != (2, 2):
        return _empty_fig(f"Formato de matriz no válido: {matrix.shape}")

    tn, fp, fn, tp = matrix.ravel()

    text = [
        [
            f"<span style='font-size:20px'><b>{tn:,}</b></span><br><span style='font-size:15px'><b>TN</b></span>",
            f"<span style='font-size:20px'><b>{fp:,}</b></span><br><span style='font-size:15px'><b>FP</b></span>",
        ],
        [
            f"<span style='font-size:20px'><b>{fn:,}</b></span><br><span style='font-size:15px'><b>FN</b></span>",
            f"<span style='font-size:20px'><b>{tp:,}</b></span><br><span style='font-size:15px'><b>TP</b></span>",
        ],
    ]

    hovertext = [
        [
            "TN: Verdaderos negativos",
            "FP: Falsos positivos",
        ],
        [
            "FN: Falsos negativos",
            "TP: Verdaderos positivos",
        ],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.astype(int),
            x=[
                "Predicción: Sin diabetes",
                "Predicción: Prediabetes / Diabetes",
            ],
            y=[
                "Real: Sin diabetes",
                "Real: Prediabetes / Diabetes",
            ],
            text=text,
            texttemplate="%{text}",
            hovertext=hovertext,
            hovertemplate="<b>%{hovertext}</b><br>Cantidad: %{z:,}<extra></extra>",
            colorscale=[
                [0.0, "#F8FBFF"],
                [0.35, "#BFE7E5"],
                [0.70, "#56DBC8"],
                [1.0, "#0F2447"],
            ],
            showscale=True,
            colorbar=dict(
                title="Casos",
                thickness=12,
                len=0.78,
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Matriz de confusión · XGBoost + Class weight · umbral {threshold:.2f}</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=17, color="#0F2447"),
        ),
        template="plotly_white",
        height=470,
        margin=dict(l=45, r=35, t=80, b=85),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="Inter, Arial, sans-serif",
            color="#0F2447",
            size=15,
        ),
    )

    fig.update_xaxes(
        side="bottom",
        tickfont=dict(size=12, color="#334E68"),
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        autorange="reversed",
        tickfont=dict(size=12, color="#334E68"),
        showgrid=False,
        zeroline=False,
    )

    return fig

# ──────────────────────────────────────────────────────────────────────────────
# 2) Curva ROC (todos los modelos o solo el mejor)
# ──────────────────────────────────────────────────────────────────────────────

def _model_color(name: str) -> str:
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    for key, color in MODEL_COLORS.items():
        if key.lower() in str(name).lower():
            return color
    return C_GRAY


def _curve_display_name(name: str, vals: dict) -> str:
    if isinstance(vals, dict):
        return vals.get("display_name") or vals.get("model_base") or name
    return name


def _selected_curve_items(data: dict, selected_models=None):
    if isinstance(selected_models, str):
        selected_models = [selected_models]
    if selected_models:
        return [(name, data[name]) for name in selected_models if name in data]
    return list(data.items())


def _fmt_auc(value) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/d"
        return f"{float(value):.3f}"
    except Exception:
        return "n/d"


def fig_class_imbalance() -> go.Figure:
    """Barra horizontal apilada con el desbalance de la variable objetivo."""
    labels = ["Sin diabetes", "Prediabetes / Diabetes"]
    counts = [218334, 35346]
    pct = [86.1, 13.9]
    colors = ["#4A7FC1", "#C0392B"]

    fig = go.Figure()
    for label, count, p, color in zip(labels, counts, pct, colors):
        fig.add_trace(go.Bar(
            y=["Variable objetivo"],
            x=[p],
            name=label,
            orientation="h",
            marker=dict(color=color, line=dict(color="rgba(15,36,71,0.12)", width=1)),
            text=[f"{p:.1f}%<br>{count:,}"],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=13, color="#0B1628", family="Inter, Arial, sans-serif"),
            hovertemplate=f"<b>{label}</b><br>Registros: {count:,}<br>Proporción: {p:.1f}%<extra></extra>",
        ))

    fig.update_layout(**_base_layout(
        title="Distribución de la variable objetivo",
        subtitle="El dataset presenta un desbalance importante: la clase positiva representa solo el 13.9% de los registros.",
        height=260,
    ))
    fig.update_layout(
        barmode="stack",
        bargap=0.55,
        margin=dict(l=18, r=18, t=78, b=42),
        legend=dict(orientation="h", x=0.5, y=-0.18, xanchor="center", borderwidth=0),
        annotations=[
            dict(
                text="<b>Razón aproximada de desbalance: 6.2:1</b>",
                x=0.985, y=1.18, xref="paper", yref="paper",
                showarrow=False, align="right",
                font=dict(size=12, color="#0F2447"),
            )
        ],
    )
    fig.update_xaxes(title_text="", range=[0, 100], ticksuffix="%", showgrid=False)
    fig.update_yaxes(title_text="", showticklabels=False)
    return fig


def fig_roc_curves(only_model: Optional[str] = None, selected_models=None) -> go.Figure:
    data = get_roc_pr_data()
    if not data:
        return _empty_fig("Datos ROC no disponibles")

    fig = go.Figure()
    selected = selected_models if selected_models is not None else only_model
    items = _selected_curve_items(data, selected)
    if not items:
        return _empty_fig("Selecciona al menos una curva para visualizar")

    for name, vals in items:
        display_name = _curve_display_name(name, vals)
        color = _model_color(display_name)
        fig.add_trace(go.Scatter(
            x=vals.get("fpr", []), y=vals.get("tpr", []),
            mode="lines",
            name=f"{display_name} (AUC = {_fmt_auc(vals.get('roc_auc'))})",
            line=dict(color=color, width=2.5),
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        ))

    # Línea diagonal de referencia (modelo aleatorio)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color=C_GRAY, width=1, dash="dash"),
        name="Aleatorio (AUC = 0.5)",
        hoverinfo="skip",
    ))

    fig.update_layout(**_base_layout(height=470))
    fig.update_xaxes(title_text="Tasa de falsos positivos (FPR)", range=[0, 1])
    fig.update_yaxes(title_text="Tasa de verdaderos positivos (TPR)", range=[0, 1.02])
    return _grid_axes(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 3) Curva Precision-Recall
# ──────────────────────────────────────────────────────────────────────────────

def fig_pr_curves(only_model: Optional[str] = None, selected_models=None) -> go.Figure:
    data = get_roc_pr_data()
    if not data:
        return _empty_fig("Datos Precision-Recall no disponibles")

    fig = go.Figure()
    selected = selected_models if selected_models is not None else only_model
    items = _selected_curve_items(data, selected)
    if not items:
        return _empty_fig("Selecciona al menos una curva para visualizar")

    for name, vals in items:
        display_name = _curve_display_name(name, vals)
        color = _model_color(display_name)
        fig.add_trace(go.Scatter(
            x=vals.get("recall", []), y=vals.get("precision", []),
            mode="lines",
            name=f"{display_name} (AP = {_fmt_auc(vals.get('pr_auc'))})",
            line=dict(color=color, width=2.5, shape="hv"),
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
        ))

    fig.update_layout(**_base_layout(height=470))
    fig.update_xaxes(title_text="Recall", range=[0, 1])
    fig.update_yaxes(title_text="Precision", range=[0, 1.02])
    return _grid_axes(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 4) Importancia de variables
# ──────────────────────────────────────────────────────────────────────────────

def fig_feature_importance(top_n: int = 12) -> go.Figure:
    """Importancia de variables con degradado azul.

    El color de cada barra interpola entre un azul muy claro (variables
    menos importantes) y el navy del dashboard (más importantes), todo
    dentro de la paleta corporativa.
    """
    df = get_feature_importance()
    if df.empty:
        return _empty_fig("Importancia de variables no disponible")

    df = df.sort_values("importance", ascending=False).head(top_n).copy()
    df["label"] = df["feature"].map(FEATURE_LABELS_ES).fillna(df["feature"])
    df = df.sort_values("importance", ascending=True)  # barras horizontales

    # ── Degradado azul: interpola entre azul claro y navy ────────────────
    def _interp_blue(t: float) -> str:
        """t en [0, 1] → color hex desde azul claro hasta navy."""
        # Punto de partida (azul claro) y de llegada (navy)
        c0 = (217, 232, 247)   # #D9E8F7  azul muy claro
        c1 = (23, 52, 95)      # #17345F  navy
        r = int(c0[0] + (c1[0] - c0[0]) * t)
        g = int(c0[1] + (c1[1] - c0[1]) * t)
        b = int(c0[2] + (c1[2] - c0[2]) * t)
        return f"rgb({r},{g},{b})"

    n = len(df)
    if n == 1:
        bar_colors = [_interp_blue(1.0)]
    else:
        bar_colors = [_interp_blue(i / (n - 1)) for i in range(n)]

    fig = go.Figure(go.Bar(
        x=df["importance"], y=df["label"],
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color="rgba(15,36,71,0.18)", width=0.6),
        ),
        text=[f"{v:.3f}" for v in df["importance"]],
        textposition="inside",
        textfont=dict(size=12, color="#0B1628", family="Inter, Arial, sans-serif"),
        hovertemplate="<b>%{y}</b><br>Importancia: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(**_base_layout())
    fig.update_xaxes(title_text="Importancia relativa")
    fig.update_yaxes(title_text="", tickfont=dict(size=12, color="#1F2D3D"))
    fig.update_layout(
        margin=dict(l=20, r=36, t=20, b=20),
        bargap=0.35,
    )
    return _grid_axes(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 5) Comparación de métricas entre modelos
# ──────────────────────────────────────────────────────────────────────────────

METRIC_LABEL_ES = {
    "accuracy":  "Accuracy",
    "precision": "Precision",
    "recall":    "Recall",
    "f1":        "F1-score",
    "roc_auc":   "ROC-AUC",
    "pr_auc":    "PR-AUC",
}


def fig_metrics_comparison(metrics: list) -> go.Figure:
    """Barras agrupadas por modelo usando la mejor combinación de balanceo."""
    df = get_metrics_df()
    if df.empty or not metrics:
        return _empty_fig("Sin métricas para comparar")

    model_order = ["Random Forest", "XGBoost", "Regresión Logística", "KNN"]
    display_names = {
        "Random Forest": "Random Forest",
        "XGBoost": "XGBoost",
        "Regresión Logística": "Logistic Regression",
        "KNN": "KNN",
    }
    best_rows = []
    for base in model_order:
        subset = df[df["model_base"] == base].copy()
        if subset.empty:
            continue
        best_rows.append(subset.sort_values("recall", ascending=False).iloc[0])

    if not best_rows:
        return _empty_fig("Sin métricas para comparar")

    plot_df = pd.DataFrame(best_rows).copy()
    plot_df["model_display"] = plot_df["model_base"].map(display_names).fillna(plot_df["model_base"])

    fig = go.Figure()
    metric_colors = {
        "precision": "#0F2447",
        "recall": "#1E3F75",
        "f1": "#2B6AAD",
        "roc_auc": "#4A7FC1",
        "accuracy": "#6FA0D6",
        "pr_auc": "#C98A16",
    }

    for metric in metrics:
        if metric not in df.columns:
            continue
        fig.add_trace(go.Bar(
            x=plot_df["model_display"],
            y=plot_df[metric],
            name=METRIC_LABEL_ES.get(metric, metric),
            marker=dict(
                color=metric_colors.get(metric, C_BLUE_L),
                line=dict(color="rgba(255,255,255,0.85)", width=1),
            ),
            text=[f"{v:.3f}" for v in plot_df[metric]],
            textposition="outside",
            textfont=dict(size=12, color="#0B1628", family="Inter, Arial, sans-serif"),
            cliponaxis=True,
            hovertemplate="<b>%{x}</b><br>"
                          + METRIC_LABEL_ES.get(metric, metric)
                          + ": %{y:.4f}<extra></extra>",
        ))

    fig.update_layout(**_base_layout())
    fig.update_layout(
        barmode="group",
        bargap=0.42,
        bargroupgap=0.12,
        margin=dict(l=72, r=30, t=78, b=66),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=12, color="#1B3A6B", family="Inter, system-ui, sans-serif"),
            itemwidth=40,
        ),
    )
    fig.update_xaxes(
        title_text="",
        tickfont=dict(size=12, color="#41536B", family="Inter, Arial, sans-serif"),
    )
    fig.update_yaxes(
        title_text="Valor de la métrica",
        title_font=dict(size=12, color="#1B3A6B"),
        range=[0, 1.08],
        tickfont=dict(size=11, color="#475569"),
    )
    return _grid_axes(fig)


def fig_recommended_model_metrics() -> go.Figure:
    """Métricas principales solo del modelo recomendado."""
    metrics = [
        ("Accuracy", 0.7207),
        ("Precision", 0.3060),
        ("Recall", 0.7920),
        ("F1-score", 0.4414),
        ("ROC-AUC", 0.8271),
        ("PR-AUC", 0.4230),
    ]
    names = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    colors = ["#4A7FC1", "#6FA0D6", "#2BB3C8", "#1B3A6B", "#0F2447", "#C98A16"]

    fig = go.Figure(go.Bar(
        x=names,
        y=values,
        marker=dict(color=colors, line=dict(color="rgba(15,36,71,0.16)", width=1)),
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Valor: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title="Métricas principales del modelo recomendado",
        subtitle="El recall se prioriza porque el objetivo es detectar la mayor cantidad posible de casos positivos.",
        height=420,
    ))
    fig.update_layout(showlegend=False, bargap=0.34, margin=dict(l=55, r=30, t=82, b=58))
    fig.update_yaxes(title_text="Valor", range=[0, 0.92])
    fig.update_xaxes(title_text="")
    fig.add_annotation(
        x="Recall", y=0.7920, text="<b>Métrica prioritaria</b>",
        showarrow=True, arrowhead=2, ax=0, ay=-42,
        font=dict(size=11, color="#0F2447"), arrowcolor="#0F2447",
    )
    return _grid_axes(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 6) Distribución de probabilidades predichas (mejor modelo)
#    Histograma de y_score separado por clase real, con línea vertical en el
#    threshold final. Muestra qué tan separadas quedan las dos clases.
# ──────────────────────────────────────────────────────────────────────────────

def fig_proba_distribution() -> go.Figure:
    """
    Histograma de las probabilidades predichas por el mejor modelo,
    discriminado por clase real (0 = sin diabetes, 1 = con diabetes).
    Incluye línea vertical en el threshold final ajustado.
    """
    import os, json
    from pathlib import Path

    # Localizar archivos
    root = Path(__file__).resolve().parents[2]
    pred_path = root / "models" / "best_model_predictions.csv"
    th_path   = root / "models" / "threshold_adjustment.json"

    if not pred_path.exists():
        return _empty_fig("No se encontró best_model_predictions.csv")

    df = pd.read_csv(pred_path)
    if "y_real" not in df.columns or "y_score" not in df.columns:
        return _empty_fig("Estructura de predicciones no reconocida")

    # Threshold final
    threshold_final = 0.5
    if th_path.exists():
        try:
            th_info = json.loads(th_path.read_text())
            threshold_final = float(th_info.get("threshold_final", 0.5))
        except Exception:
            pass

    scores_neg = df.loc[df["y_real"] == 0, "y_score"].values
    scores_pos = df.loc[df["y_real"] == 1, "y_score"].values

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores_neg, name="Sin diabetes (real = 0)",
        marker_color=C_BLUE_L, opacity=0.78,
        nbinsx=50, histnorm="probability density",
        hovertemplate="Probabilidad: %{x:.2f}<br>Densidad: %{y:.2f}<extra>Sin diabetes</extra>",
    ))
    fig.add_trace(go.Histogram(
        x=scores_pos, name="Con diabetes (real = 1)",
        marker_color=C_RED, opacity=0.78,
        nbinsx=50, histnorm="probability density",
        hovertemplate="Probabilidad: %{x:.2f}<br>Densidad: %{y:.2f}<extra>Con diabetes</extra>",
    ))

    # Línea del threshold ajustado
    fig.add_vline(
        x=threshold_final, line_dash="dash", line_color="#0B1628", line_width=2,
        annotation_text=f"Threshold = {threshold_final:.3f}",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#0B1628"),
    )

    fig.update_layout(**_base_layout(height=380))
    fig.update_layout(
        barmode="overlay",
        legend=dict(orientation="h", x=0.5, y=1.08, xanchor="center"),
        xaxis_title="Probabilidad predicha de tener diabetes",
        yaxis_title="Densidad",
    )
    return _grid_axes(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 7) Trade-off de métricas vs threshold (mejor modelo)
#    Curvas de Precision, Recall y F1 al variar el umbral de decisión, con
#    línea vertical en el threshold final.
# ──────────────────────────────────────────────────────────────────────────────

def fig_threshold_tradeoff() -> go.Figure:
    """
    Curvas Precision / Recall / F1 calculadas al variar el threshold sobre las
    predicciones del mejor modelo, con línea vertical en el threshold final.
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    pred_path = root / "models" / "best_model_predictions.csv"
    th_path   = root / "models" / "threshold_adjustment.json"

    if not pred_path.exists():
        return _empty_fig("No se encontró best_model_predictions.csv")

    df = pd.read_csv(pred_path)
    y_true   = df["y_real"].values
    y_score  = df["y_score"].values

    # Recorrido de thresholds 0.05 a 0.95
    thresholds = np.linspace(0.05, 0.95, 91)
    precisions, recalls, f1s = [], [], []

    pos_mask = y_true == 1
    n_pos = pos_mask.sum()

    for th in thresholds:
        y_pred = (y_score >= th).astype(int)
        tp = int(((y_pred == 1) & pos_mask).sum())
        fp = int(((y_pred == 1) & ~pos_mask).sum())
        fn = int(((y_pred == 0) & pos_mask).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / n_pos if n_pos else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        precisions.append(prec); recalls.append(rec); f1s.append(f1)

    threshold_final = 0.5
    if th_path.exists():
        try:
            th_info = json.loads(th_path.read_text())
            threshold_final = float(th_info.get("threshold_final", 0.5))
        except Exception:
            pass

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds, y=recalls, mode="lines", name="Recall",
        line=dict(color=C_RED, width=3),
        hovertemplate="Threshold: %{x:.2f}<br>Recall: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=precisions, mode="lines", name="Precision",
        line=dict(color=C_BLUE, width=3),
        hovertemplate="Threshold: %{x:.2f}<br>Precision: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=f1s, mode="lines", name="F1-score",
        line=dict(color="#C98A16", width=3, dash="dot"),
        hovertemplate="Threshold: %{x:.2f}<br>F1: %{y:.3f}<extra></extra>",
    ))

    fig.add_vline(
        x=threshold_final, line_dash="dash", line_color="#0B1628", line_width=2,
        annotation_text=f"Threshold óptimo = {threshold_final:.3f}",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#0B1628"),
    )
    fig.add_vline(
        x=0.5, line_dash="dot", line_color=C_GRAY, line_width=1.5,
        annotation_text="Default 0.50",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color=C_GRAY),
    )

    fig.update_layout(**_base_layout(height=380))
    fig.update_layout(
        legend=dict(orientation="h", x=0.5, y=1.08, xanchor="center"),
        xaxis_title="Threshold de decisión",
        yaxis_title="Valor de la métrica",
    )
    fig.update_yaxes(range=[0, 1.05])
    return _grid_axes(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 8) Heatmap de Recall por (Modelo base × Estrategia de balanceo)
#    Sirve para mostrar de un vistazo cuál combinación dio mejor recall.
# ──────────────────────────────────────────────────────────────────────────────

def fig_three_metric_heatmaps() -> go.Figure:
    """Tres heatmaps compactos: Recall, F1-score y ROC-AUC."""
    df = get_metrics_df()
    if df is None or df.empty:
        return _empty_fig("Sin métricas disponibles")

    if "model_base" not in df.columns or "balance_strategy" not in df.columns:
        return _empty_fig("Estructura de métricas no reconocida")

    model_order = ["Regresión Logística", "Random Forest", "XGBoost", "KNN"]
    balance_order = ["none", "class_weight", "smote", "adasyn"]
    balance_labels = {
        "none": "Sin balanceo",
        "class_weight": "Class weight",
        "smote": "SMOTE",
        "adasyn": "ADASYN",
    }
    metric_specs = [
        ("recall", "Recall", [[0, "#EAF8F8"], [0.45, "#9EDFE4"], [1, "#2BB3C8"]]),
        ("f1", "F1-score", [[0, "#EEF5FC"], [0.55, "#8DB8E2"], [1, "#4A7FC1"]]),
        ("roc_auc", "ROC-AUC", [[0, "#EEF3F8"], [0.55, "#6E88AB"], [1, "#0F2447"]]),
    ]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[spec[1] for spec in metric_specs],
        horizontal_spacing=0.055,
    )

    for i, (metric, label, scale) in enumerate(metric_specs, start=1):
        pivot = df.pivot_table(
            index="model_base",
            columns="balance_strategy",
            values=metric,
            aggfunc="max",
        ).reindex(index=model_order, columns=balance_order)

        z = pivot.values.astype(float)
        text = [[f"{v:.3f}" if pd.notna(v) else "-" for v in row] for row in z]
        hover = [
            [
                f"Modelo: {model}<br>Balanceo: {balance_labels[col]}<br>{label}: {val:.3f}"
                if pd.notna(val)
                else f"Modelo: {model}<br>Balanceo: {balance_labels[col]}<br>No aplica"
                for col, val in zip(balance_order, row)
            ]
            for model, row in zip(model_order, z)
        ]

        finite = z[np.isfinite(z)]
        zmin = max(0, float(finite.min()) - 0.02) if finite.size else 0
        zmax = min(1, float(finite.max()) + 0.02) if finite.size else 1

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=[balance_labels[b] for b in balance_order],
                y=model_order,
                colorscale=scale,
                zmin=zmin,
                zmax=zmax,
                text=text,
                texttemplate="%{text}",
                textfont=dict(size=12, color="#0B1628", family="Inter, Arial, sans-serif"),
                hovertext=hover,
                hovertemplate="%{hovertext}<extra></extra>",
                showscale=False,
                xgap=4,
                ygap=4,
            ),
            row=1,
            col=i,
        )

    fig.update_layout(**_base_layout(
        title="Desempeño por modelo y técnica de balanceo",
        subtitle="Comparación de las tres métricas principales usadas para evaluar el comportamiento de las combinaciones entrenadas.",
        height=430,
    ))
    fig.update_layout(
        margin=dict(l=24, r=24, t=92, b=80),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    fig.update_annotations(font=dict(size=13, color="#0F2447", family="Inter, Arial, sans-serif"))
    for col in range(1, 4):
        fig.update_xaxes(tickangle=-25, side="bottom", tickfont=dict(size=10), row=1, col=col)
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=11), row=1, col=col)
    return fig


def fig_recall_heatmap() -> go.Figure:
    """Alias de compatibilidad: ahora muestra los tres heatmaps principales."""
    return fig_three_metric_heatmaps()
