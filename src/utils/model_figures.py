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

from src.utils.model_utils import (
    FEATURE_LABELS_ES,
    get_feature_importance,
    get_metrics_df,
    get_roc_pr_data,
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
    "KNN":                  "#C98A16",  # ámbar
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

def fig_confusion_matrix(cm_data: Optional[dict]) -> go.Figure:
    if not cm_data or "matrix" not in cm_data:
        return _empty_fig("Matriz de confusión no disponible")

    cm = np.asarray(cm_data["matrix"])
    labels = ["Sin diabetes", "Prediabetes / Diabetes"]

    # Texto en cada celda con conteo y porcentaje por fila
    row_totals = cm.sum(axis=1, keepdims=True)
    pct = np.where(row_totals > 0, cm / row_totals * 100, 0)
    text = [[f"<b>{cm[i, j]:,}</b><br>{pct[i, j]:.1f}%"
             for j in range(cm.shape[1])] for i in range(cm.shape[0])]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels, y=labels,
        text=text, texttemplate="%{text}",
        textfont=dict(size=14, color="#0B1628"),
        colorscale=[[0, "#E8EFF8"], [0.5, "#88AED8"], [1, C_BLUE]],
        showscale=False,
        hovertemplate="Real: <b>%{y}</b><br>Predicho: <b>%{x}</b><br>Casos: %{z:,}<extra></extra>",
    ))
    fig.update_layout(**_base_layout())
    fig.update_xaxes(title_text="Predicción del modelo", side="bottom")
    fig.update_yaxes(title_text="Clase real", autorange="reversed")
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 2) Curva ROC (todos los modelos o solo el mejor)
# ──────────────────────────────────────────────────────────────────────────────

def fig_roc_curves(only_model: Optional[str] = None) -> go.Figure:
    data = get_roc_pr_data()
    if not data:
        return _empty_fig("Datos ROC no disponibles")

    fig = go.Figure()
    items = data.items() if only_model is None else [
        (only_model, data[only_model])
    ] if only_model in data else []

    for name, vals in items:
        color = MODEL_COLORS.get(name, C_GRAY)
        fig.add_trace(go.Scatter(
            x=vals["fpr"], y=vals["tpr"],
            mode="lines",
            name=f"{name} (AUC = {vals['roc_auc']:.3f})",
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

    fig.update_layout(**_base_layout())
    fig.update_xaxes(title_text="Tasa de falsos positivos (FPR)", range=[0, 1])
    fig.update_yaxes(title_text="Tasa de verdaderos positivos (TPR)", range=[0, 1.02])
    return _grid_axes(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 3) Curva Precision-Recall
# ──────────────────────────────────────────────────────────────────────────────

def fig_pr_curves(only_model: Optional[str] = None) -> go.Figure:
    data = get_roc_pr_data()
    if not data:
        return _empty_fig("Datos Precision-Recall no disponibles")

    fig = go.Figure()
    items = data.items() if only_model is None else [
        (only_model, data[only_model])
    ] if only_model in data else []

    for name, vals in items:
        color = MODEL_COLORS.get(name, C_GRAY)
        fig.add_trace(go.Scatter(
            x=vals["recall"], y=vals["precision"],
            mode="lines",
            name=f"{name} (AP = {vals['pr_auc']:.3f})",
            line=dict(color=color, width=2.5, shape="hv"),
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
        ))

    fig.update_layout(**_base_layout())
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
    """Barras agrupadas: una métrica seleccionada por barra, un modelo por
    posición. Si se pasan varias métricas, las agrupa.
    """
    df = get_metrics_df()
    if df.empty or not metrics:
        return _empty_fig("Sin métricas para comparar")

    # Mantener el orden por F1 desc
    fig = go.Figure()
    # Paleta solo en tonalidades azules (oscuro -> claro)
    blue_palette = [
        "#0F2447",  # navy
        "#1B3A6B",  # azul oscuro
        "#2563A6",  # azul medio
        "#4A7FC1",  # azul
        "#6FA0D6",  # azul claro
        "#9BC0E2",  # azul muy claro
    ]

    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            continue
        fig.add_trace(go.Bar(
            x=df["model"],
            y=df[metric],
            name=METRIC_LABEL_ES.get(metric, metric),
            marker=dict(
                color=blue_palette[i % len(blue_palette)],
                line=dict(color="rgba(255,255,255,0.85)", width=1),
            ),
            text=[f"{v:.3f}" for v in df[metric]],
            textposition="inside",
            textfont=dict(size=12, color="#0B1628", family="Inter, system-ui, sans-serif"),
            cliponaxis=True,
            hovertemplate="<b>%{x}</b><br>"
                          + METRIC_LABEL_ES.get(metric, metric)
                          + ": %{y:.4f}<extra></extra>",
        ))

    fig.update_layout(**_base_layout())
    fig.update_layout(
        barmode="group",
        bargap=0.38,
        bargroupgap=0.18,
        margin=dict(l=60, r=30, t=70, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
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
        tickfont=dict(size=12, color="#0B1628", family="Inter, system-ui, sans-serif"),
    )
    fig.update_yaxes(
        title_text="Valor de la métrica",
        title_font=dict(size=12, color="#1B3A6B"),
        range=[0, 1.08],
        tickfont=dict(size=11, color="#475569"),
    )
    return _grid_axes(fig)
