import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.data_loader import (
    AGE_ORDER, GENHLTH_ORDER, EDUCATION_ORDER, INCOME_ORDER,
    BINARY_MAPPINGS, FACTOR_OPTIONS,
)

# ── Paleta profesional ────────────────────────────────────────────────────────
C_BLUE   = "#17345F"
C_RED    = "#B83A32"
C_BLUE_L = "#4A7FC1"
C_GRAY   = "#64748B"
C_BG     = "#FFFFFF"
C_GRID   = "#E5EAF2"

COLORS = {"Sin diabetes": C_BLUE, "Prediabetes / Diabetes": C_RED}

_LEGEND_DEFAULT = dict(
    bgcolor="rgba(255,255,255,0.92)",
    bordercolor=C_GRID,
    borderwidth=1,
    font=dict(size=11, color="#334155"),
)


def _base_layout(title, subtitle="", extra=None):
    full_title = f"<b>{title}</b>"
    if subtitle:
        full_title += (
            f"<br><span style='font-size:11px;color:{C_GRAY}'>{subtitle}</span>"
        )
    layout = dict(
        template="plotly_white",
        font=dict(family="Inter, Arial, sans-serif", size=12, color="#334155"),
        paper_bgcolor=C_BG,
        plot_bgcolor=C_BG,
        margin=dict(l=18, r=18, t=70, b=18),
        title=dict(text=full_title, x=0.02, xanchor="left", font=dict(size=14, color="#0B1628")),
        legend=_LEGEND_DEFAULT,
        hoverlabel=dict(
            bgcolor="#0B1628",
            bordercolor="#0B1628",
            font=dict(color="#FFFFFF", size=12),
        ),
    )
    if extra:
        layout.update(extra)
    return layout


def _grid_axes(fig):
    fig.update_xaxes(
        gridcolor=C_GRID,
        zerolinecolor=C_GRID,
        linecolor="#CBD5E1",
        tickfont=dict(color="#475569"),
        title_font=dict(color="#334155"),
    )
    fig.update_yaxes(
        gridcolor=C_GRID,
        zerolinecolor=C_GRID,
        linecolor="#CBD5E1",
        tickfont=dict(color="#475569"),
        title_font=dict(color="#334155"),
    )
    return fig


def _clean_legend_bottom(y=-0.24):
    return dict(
        orientation="h",
        yanchor="top",
        y=y,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(255,255,255,0)",
        borderwidth=0,
        font=dict(size=11, color="#334155"),
    )


def _clean_bar_traces(fig):
    fig.update_traces(
        text=None,
        texttemplate=None,
        cliponaxis=False,
        selector=dict(type="bar"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1. DISTRIBUCIÓN DIABETES — donut
# ─────────────────────────────────────────────────────────────────────────────

def fig_diabetes_distribution(df_viz):
    if df_viz.empty:
        return go.Figure()

    counts = df_viz["Diabetes_binary"].value_counts().reset_index()
    counts.columns = ["Estado", "Cantidad"]
    total = counts["Cantidad"].sum()
    counts["Pct"] = (counts["Cantidad"] / total * 100).round(1)

    diab_row = counts[counts["Estado"] == "Prediabetes / Diabetes"]
    center_pct = f"{diab_row['Pct'].values[0]}%" if len(diab_row) > 0 else ""

    fig = go.Figure(go.Pie(
        labels=counts["Estado"],
        values=counts["Cantidad"],
        hole=0.60,
        sort=False,
        marker_colors=[COLORS.get(e, C_GRAY) for e in counts["Estado"]],
        textinfo="percent",
        textposition="inside",
        insidetextorientation="radial",
        textfont=dict(size=13, color="white"),
        hovertemplate="<b>%{label}</b><br>%{value:,} individuos<br>%{percent}<extra></extra>",
        pull=[0.03 if e == "Prediabetes / Diabetes" else 0 for e in counts["Estado"]],
    ))

    fig.update_layout(**_base_layout(
        "Distribución de diabetes",
        "Composición de la muestra",
        extra=dict(
            annotations=[dict(
                text=(
                    f"<b>{center_pct}</b>"
                    f"<br><span style='font-size:10px;color:{C_GRAY}'>diabetes</span>"
                ),
                x=0.5, y=0.5, font_size=18, showarrow=False,
                font=dict(color=C_RED),
            )],
            showlegend=True,
            legend=_clean_legend_bottom(-0.08),
            height=340,
            margin=dict(l=18, r=18, t=76, b=62),
            uniformtext=dict(minsize=11, mode="hide"),
        )
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. BMI HISTOGRAMA (densidad superpuesta) — usado en dashboard
# ─────────────────────────────────────────────────────────────────────────────

def fig_bmi_histogram(df_viz):
    if df_viz.empty or "BMI" not in df_viz.columns:
        return go.Figure()

    dm     = df_viz["Diabetes_binary"] == "Prediabetes / Diabetes"
    bmi_dm = df_viz[dm]["BMI"].dropna()
    bmi_no = df_viz[~dm]["BMI"].dropna()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=bmi_no, name="Sin diabetes",
        marker_color=C_BLUE, opacity=0.65,
        nbinsx=50, histnorm="probability density",
        hovertemplate="IMC: %{x:.1f}<br>Densidad: %{y:.4f}<extra>Sin diabetes</extra>",
    ))
    fig.add_trace(go.Histogram(
        x=bmi_dm, name="Prediabetes / Diabetes",
        marker_color=C_RED, opacity=0.65,
        nbinsx=50, histnorm="probability density",
        hovertemplate="IMC: %{x:.1f}<br>Densidad: %{y:.4f}<extra>Con diabetes</extra>",
    ))

    if len(bmi_no) > 0:
        fig.add_vline(
            x=bmi_no.mean(), line_dash="dash", line_color=C_BLUE, line_width=1.5,
        )
    if len(bmi_dm) > 0:
        fig.add_vline(
            x=bmi_dm.mean(), line_dash="dash", line_color=C_RED, line_width=1.5,
        )

    bmi_max = df_viz["BMI"].max()
    if bmi_max > 30:
        fig.add_vrect(
            x0=30, x1=bmi_max,
            fillcolor=C_RED, opacity=0.04, line_width=0,
        )

    fig.update_layout(**_base_layout(
        "IMC por estado de diabetes",
        "Histograma de densidad",
        extra=dict(
            barmode="overlay",
            height=340,
            legend=_clean_legend_bottom(-0.22),
            margin=dict(l=18, r=18, t=76, b=74),
        )
    ))
    fig.update_xaxes(title_text="Índice de Masa Corporal (IMC)",
                     gridcolor=C_GRID, linecolor="#CBD5E1", automargin=True)
    fig.update_yaxes(title_text="Densidad",
                     gridcolor=C_GRID, linecolor="#CBD5E1", automargin=True)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREVALENCIA POR EDAD — línea + barras volumen
# ─────────────────────────────────────────────────────────────────────────────

def fig_age_distribution(df_viz):
    if df_viz.empty or "Age" not in df_viz.columns:
        return go.Figure()

    order = [a for a in AGE_ORDER if a in df_viz["Age"].cat.categories]
    if not order:
        return go.Figure()

    prev_by_age = (
        df_viz.groupby("Age", observed=True)
        .apply(lambda g: (g["Diabetes_binary"] == "Prediabetes / Diabetes").mean() * 100)
        .reindex(order)
        .reset_index()
    )
    prev_by_age.columns = ["Age", "Prevalencia (%)"]
    vol = df_viz.groupby("Age", observed=True).size().reindex(order)
    prev_by_age["N"] = vol.values

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=prev_by_age["Age"],
        y=prev_by_age["N"],
        name="Volumen (N)",
        marker_color=C_BLUE_L,
        opacity=0.25,
        yaxis="y2",
        hovertemplate="%{x}<br>N = %{y:,}<extra>Volumen</extra>",
    ))

    fig.add_trace(go.Scatter(
        x=prev_by_age["Age"],
        y=prev_by_age["Prevalencia (%)"],
        name="Prevalencia (%)",
        mode="lines+markers",
        line=dict(color=C_RED, width=2.5),
        marker=dict(size=7, color=C_RED),
        hovertemplate="%{x}<br>Prevalencia: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(**_base_layout(
        "Prevalencia por edad",
        "Porcentaje con diabetes y volumen muestral",
        extra=dict(
            height=340,
            hovermode="x unified",
            legend=_clean_legend_bottom(-0.24),
            margin=dict(l=18, r=18, t=76, b=86),
            yaxis=dict(
                title="Prevalencia (%)", range=[0, 50],
                ticksuffix="%", gridcolor=C_GRID, automargin=True,
            ),
            yaxis2=dict(
                title="", overlaying="y", side="right",
                showgrid=False, showticklabels=False,
            ),
            xaxis=dict(title="", tickangle=-35, gridcolor=C_GRID, automargin=True),
        )
    ))
    
    fig.update_xaxes(
        tickangle=0,
        automargin=True,
        tickfont=dict(size=11)
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. FACTOR vs DIABETES — panel interactivo del Dashboard
#    Binarias → barras agrupadas | Ordinales → barras apiladas 100%
# ─────────────────────────────────────────────────────────────────────────────

def _get_factor_order(factor):
    return {
        "GenHlth":   GENHLTH_ORDER,
        "Age":       AGE_ORDER,
        "Education": EDUCATION_ORDER,
        "Income":    INCOME_ORDER,
    }.get(factor)


_ORDINAL_SET = {"GenHlth", "Age", "Education", "Income"}


def fig_factor_vs_diabetes(df_viz, factor):
    if not factor or factor not in df_viz.columns or df_viz.empty:
        return go.Figure()

    factor_label = next(
        (o["label"] for o in FACTOR_OPTIONS if o["value"] == factor), factor
    )
    order = _get_factor_order(factor)
    cross_abs = pd.crosstab(df_viz[factor], df_viz["Diabetes_binary"])

    if order:
        existing = [o for o in order if o in cross_abs.index]
        if existing:
            cross_abs = cross_abs.reindex(existing)

    if factor in _ORDINAL_SET:
        return _fig_stacked_100(
            cross_abs, factor_label,
            f"Diabetes según {factor_label}",
        )
    else:
        return _fig_grouped_bars(
            cross_abs, factor_label,
            f"Diabetes según {factor_label}",
        )


def _fig_stacked_100(cross_abs, factor_label, title):
    totals    = cross_abs.sum(axis=1)
    cross_pct = cross_abs.div(totals, axis=0) * 100

    fig    = go.Figure()
    no_col = "Sin diabetes"
    si_col = "Prediabetes / Diabetes"

    if no_col in cross_pct.columns:
        vals = cross_pct[no_col].round(1)
        ns   = cross_abs[no_col] if no_col in cross_abs.columns else pd.Series(dtype=int)
        fig.add_trace(go.Bar(
            x=cross_pct.index.astype(str),
            y=vals,
            name="Sin diabetes",
            marker_color=C_BLUE,
            hovertemplate="%{x}<br>Sin diabetes: %{y:.1f}%<br>N=%{customdata:,}<extra></extra>",
            customdata=ns.values if len(ns) else [],
        ))

    if si_col in cross_pct.columns:
        vals = cross_pct[si_col].round(1)
        ns   = cross_abs[si_col] if si_col in cross_abs.columns else pd.Series(dtype=int)
        fig.add_trace(go.Bar(
            x=cross_pct.index.astype(str),
            y=vals,
            name="Prediabetes / Diabetes",
            marker_color=C_RED,
            hovertemplate="%{x}<br>Con diabetes: %{y:.1f}%<br>N=%{customdata:,}<extra></extra>",
            customdata=ns.values if len(ns) else [],
        ))

    fig.update_layout(**_base_layout(
        title,
        "Distribución porcentual por categoría",
        extra=dict(
            barmode="stack",
            height=380,
            margin=dict(l=18, r=18, t=76, b=92),
            xaxis=dict(title="", tickangle=-25, gridcolor=C_GRID, automargin=True),
            yaxis=dict(
                title="Porcentaje (%)", range=[0, 105],
                ticksuffix="%", gridcolor=C_GRID, automargin=True,
            ),
            legend=_clean_legend_bottom(-0.28),
        )
    ))
    _clean_bar_traces(fig)
    return fig


def _fig_grouped_bars(cross_abs, factor_label, title):
    totals    = cross_abs.sum(axis=1)
    cross_pct = cross_abs.div(totals, axis=0) * 100

    dm_order = ["Sin diabetes", "Prediabetes / Diabetes"]
    cross_abs = cross_abs.reindex([c for c in dm_order if c in cross_abs.index])
    cross_pct = cross_pct.reindex([c for c in dm_order if c in cross_pct.index])

    palette = [C_BLUE, C_RED, "#C98A16", "#0F766E", "#64748B"]
    fig = go.Figure()

    for i, cat in enumerate(cross_abs.columns):
        n_vals   = cross_abs[cat].values
        pct_vals = cross_pct[cat].round(1).values if cat in cross_pct.columns else [0]*len(n_vals)
        color    = palette[i % len(palette)]

        fig.add_trace(go.Bar(
            x=[str(c) for c in cross_abs.index],
            y=n_vals,
            name=str(cat),
            marker_color=color,
            customdata=pct_vals,
            hovertemplate=(
                f"<b>{cat}</b><br>"
                "Grupo: %{x}<br>"
                "N: %{y:,}<br>"
                "Porcentaje del grupo: %{customdata:.1f}%<extra></extra>"
            ),
        ))

    fig.update_layout(**_base_layout(
        title,
        "Conteo por grupo de diabetes",
        extra=dict(
            barmode="group",
            height=380,
            bargap=0.28,
            xaxis=dict(title="", gridcolor=C_GRID, automargin=True),
            yaxis=dict(title="Número de personas", gridcolor=C_GRID, tickformat=",", automargin=True),
            legend=_clean_legend_bottom(-0.22),
            margin=dict(l=18, r=18, t=76, b=82),
        )
    ))
    _clean_bar_traces(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. BOXPLOT BMI (dashboard)
# ─────────────────────────────────────────────────────────────────────────────

def fig_bmi_boxplot(df_viz):
    if df_viz.empty or "BMI" not in df_viz.columns:
        return go.Figure()

    fig = go.Figure()
    for label, color in [("Sin diabetes", C_BLUE), ("Prediabetes / Diabetes", C_RED)]:
        sub = df_viz[df_viz["Diabetes_binary"] == label]["BMI"].dropna()
        if len(sub) == 0:
            continue
        fig.add_trace(go.Box(
            y=sub,
            name=label,
            marker_color=color,
            line_color=color,
            fillcolor=color,
            opacity=0.75,
            boxmean=True,
            whiskerwidth=0.5,
            hovertemplate=f"<b>{label}</b><br>IMC: %{{y:.1f}}<extra></extra>",
        ))

    fig.add_hline(
        y=30, line_dash="dot", line_color=C_GRAY, line_width=1,
    )

    fig.update_layout(**_base_layout(
        "IMC por estado de diabetes",
        "Mediana, dispersión y media",
        extra=dict(
            height=340,
            showlegend=False,
            margin=dict(l=18, r=18, t=76, b=38),
            yaxis=dict(title="Índice de Masa Corporal (IMC)", gridcolor=C_GRID, automargin=True),
            xaxis=dict(title="", gridcolor=C_GRID, automargin=True),
        )
    ))
    _grid_axes(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. FACTOR RANKING — barras divergentes
# ─────────────────────────────────────────────────────────────────────────────

def fig_factor_ranking(df_rank):
    if df_rank is None or df_rank.empty:
        return go.Figure()

    df_plot = df_rank.sort_values("Diferencia (pp)", ascending=True).copy()
    colors  = [C_RED if d > 0 else C_BLUE for d in df_plot["Diferencia (pp)"]]

    fig = go.Figure(go.Bar(
        y=df_plot["Factor"],
        x=df_plot["Diferencia (pp)"],
        orientation="h",
        marker_color=colors,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Con DM: %{customdata[0]:.1f}%<br>"
            "Sin DM: %{customdata[1]:.1f}%<br>"
            "Diferencia: %{x:+.1f} pp<extra></extra>"
        ),
        customdata=df_plot[["Prevalencia con DM", "Prevalencia sin DM"]].values,
    ))

    fig.add_vline(x=0, line_color=C_GRAY, line_width=1)

    fig.update_layout(**_base_layout(
        "Ranking de factores asociados",
        "Diferencia de prevalencia en puntos porcentuales",
        extra=dict(
            height=420,
            showlegend=False,
            xaxis=dict(
                title="Diferencia (pp)", ticksuffix=" pp",
                zeroline=True, gridcolor=C_GRID, automargin=True,
            ),
            yaxis=dict(title="", gridcolor=C_GRID, automargin=True),
            margin=dict(l=18, r=18, t=76, b=38),
        )
    ))
    _clean_bar_traces(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. HEATMAP DE CORRELACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def fig_correlation_heatmap(df_viz):
    if df_viz.empty:
        return go.Figure()

    df_corr = pd.DataFrame({
        "IMC (BMI)":     df_viz["BMI"],
        "Salud mental":  df_viz["MentHlth"],
        "Salud fisica":  df_viz["PhysHlth"],
        "Salud general": df_viz["GenHlth"].cat.codes + 1,
        "Edad":          df_viz["Age"].cat.codes + 1,
        "Educacion":     df_viz["Education"].cat.codes + 1,
        "Ingreso":       df_viz["Income"].cat.codes + 1,
    }).dropna()

    if df_corr.empty:
        return go.Figure()

    corr = df_corr.corr(method="pearson").round(2)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, C_BLUE], [0.5, "#FFFFFF"], [1, C_RED]],
        zmid=0, zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="<b>%{text}</b>",
        textfont=dict(size=10),
        hoverongaps=False,
        hovertemplate="%{y} / %{x}<br>r = %{z:.2f}<extra></extra>",
        colorbar=dict(title="r", tickvals=[-1, -0.5, 0, 0.5, 1], thickness=12, len=0.78),
    ))

    fig.update_layout(**_base_layout(
        "Correlaciones entre indicadores",
        "Coeficiente de Pearson",
        extra=dict(
            height=440,
            showlegend=False,
            margin=dict(l=18, r=18, t=76, b=92),
            xaxis=dict(tickangle=-30, automargin=True),
            yaxis=dict(automargin=True),
        ),
    ))
    
    #fig.update_layout(
        #width=1300,
        #height=500
    #)
    
    fig.update_xaxes(tickangle=0)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# ANÁLISIS BIVARIADO — Sección Analysis (About)
# Tres familias de visualizaciones según tipo de variable
# ═════════════════════════════════════════════════════════════════════════════

BINARY_VARS = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk",
]
ORDINAL_VARS_BI = ["GenHlth", "Age", "Education", "Income"]
NUMERIC_VARS_BI = ["BMI", "MentHlth", "PhysHlth"]

VAR_LABELS = {
    "HighBP":               "Hipertensión",
    "HighChol":             "Colesterol alto",
    "CholCheck":            "Chequeo de colesterol",
    "Smoker":               "Tabaquismo",
    "Stroke":               "ACV previo",
    "HeartDiseaseorAttack": "Enfermedad cardíaca",
    "PhysActivity":         "Actividad física",
    "Fruits":               "Consumo de frutas",
    "Veggies":              "Consumo de verduras",
    "HvyAlcoholConsump":    "Consumo alto de alcohol",
    "AnyHealthcare":        "Cobertura médica",
    "NoDocbcCost":          "Barrera económica de acceso",
    "DiffWalk":             "Dificultad al caminar",
    "GenHlth":              "Salud general percibida",
    "Age":                  "Grupo de edad",
    "Education":            "Nivel educativo",
    "Income":               "Nivel de ingreso",
    "BMI":                  "Índice de masa corporal (IMC)",
    "MentHlth":             "Días de mala salud mental",
    "PhysHlth":             "Días de mala salud física",
}

BIVARIATE_OPTIONS = (
    [{"label": VAR_LABELS[v], "value": v} for v in BINARY_VARS] +
    [{"label": VAR_LABELS[v], "value": v} for v in ORDINAL_VARS_BI] +
    [{"label": VAR_LABELS[v], "value": v} for v in NUMERIC_VARS_BI]
)


def _bi_base_layout(title, subtitle="", extra=None):
    full_title = f"<b>{title}</b>"
    if subtitle:
        full_title += (
            f"<br><span style='font-size:11px;color:{C_GRAY}'>{subtitle}</span>"
        )
    layout = dict(
        template="plotly_white",
        font=dict(family="Inter, Arial, sans-serif", size=12, color="#334155"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=18, r=18, t=72, b=18),
        title=dict(text=full_title, x=0.02, xanchor="left", font=dict(size=13, color="#0B1628")),
        legend=dict(
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor=C_GRID,
            borderwidth=1,
            font=dict(size=11, color="#334155"),
        ),
        hoverlabel=dict(
            bgcolor="#0B1628",
            bordercolor="#0B1628",
            font=dict(color="#FFFFFF", size=12),
        ),
    )
    if extra:
        layout.update(extra)
    return layout


def fig_bivariate_binary(df_viz, var):
    """Barras agrupadas para variable binaria vs Diabetes_binary."""
    if df_viz.empty or var not in df_viz.columns:
        return go.Figure()

    label = VAR_LABELS.get(var, var)
    cross_abs = pd.crosstab(df_viz["Diabetes_binary"], df_viz[var])
    totals = cross_abs.sum(axis=1)
    cross_pct = cross_abs.div(totals, axis=0) * 100

    dm_order = ["Sin diabetes", "Prediabetes / Diabetes"]
    cross_abs = cross_abs.reindex([c for c in dm_order if c in cross_abs.index])
    cross_pct = cross_pct.reindex([c for c in dm_order if c in cross_pct.index])

    palette = [C_BLUE, C_RED, "#C98A16", "#0F766E", "#64748B"]
    fig = go.Figure()

    for i, cat in enumerate(cross_abs.columns):
        n_vals   = cross_abs[cat].values
        pct_vals = cross_pct[cat].round(1).values if cat in cross_pct.columns else [0]*len(n_vals)
        color    = palette[i % len(palette)]

        fig.add_trace(go.Bar(
            x=[str(c) for c in cross_abs.index],
            y=n_vals,
            name=str(cat),
            marker_color=color,
            customdata=pct_vals,
            hovertemplate=(
                f"<b>{cat}</b><br>"
                "Grupo: %{x}<br>"
                "N: %{y:,}<br>"
                "Porcentaje del grupo: %{customdata:.1f}%<extra></extra>"
            ),
        ))

    fig.update_layout(**_bi_base_layout(
        f"Diabetes según {label}",
        "Conteo por grupo de diabetes",
        extra=dict(
            barmode="group",
            height=390,
            bargap=0.28,
            xaxis=dict(title="", gridcolor=C_GRID, automargin=True),
            yaxis=dict(title="Número de personas", gridcolor=C_GRID, tickformat=",", automargin=True),
            legend=_clean_legend_bottom(-0.22),
            margin=dict(l=18, r=18, t=76, b=82),
        )
    ))
    _clean_bar_traces(fig)
    return fig


def fig_bivariate_ordinal(df_viz, var):
    """Barras apiladas al 100% para variable ordinal vs Diabetes_binary."""
    if df_viz.empty or var not in df_viz.columns:
        return go.Figure()

    label = VAR_LABELS.get(var, var)
    order = _get_factor_order(var)

    cross_abs = pd.crosstab(df_viz[var], df_viz["Diabetes_binary"])
    if order:
        existing = [o for o in order if o in cross_abs.index]
        if existing:
            cross_abs = cross_abs.reindex(existing)

    totals    = cross_abs.sum(axis=1)
    cross_pct = cross_abs.div(totals, axis=0) * 100

    fig    = go.Figure()
    no_col = "Sin diabetes"
    si_col = "Prediabetes / Diabetes"

    if no_col in cross_pct.columns:
        vals = cross_pct[no_col].round(1)
        ns   = cross_abs[no_col] if no_col in cross_abs.columns else pd.Series(dtype=int)
        fig.add_trace(go.Bar(
            x=cross_pct.index.astype(str),
            y=vals,
            name="Sin diabetes",
            marker_color=C_BLUE,
            hovertemplate="%{x}<br>Sin diabetes: %{y:.1f}%<br>N=%{customdata:,}<extra></extra>",
            customdata=ns.values if len(ns) else [],
        ))

    if si_col in cross_pct.columns:
        vals = cross_pct[si_col].round(1)
        ns   = cross_abs[si_col] if si_col in cross_abs.columns else pd.Series(dtype=int)
        fig.add_trace(go.Bar(
            x=cross_pct.index.astype(str),
            y=vals,
            name="Prediabetes / Diabetes",
            marker_color=C_RED,
            hovertemplate="%{x}<br>Con diabetes: %{y:.1f}%<br>N=%{customdata:,}<extra></extra>",
            customdata=ns.values if len(ns) else [],
        ))

    fig.update_layout(**_bi_base_layout(
        f"Diabetes según {label}",
        "Distribución porcentual por categoría",
        extra=dict(
            barmode="stack",
            height=390,
            margin=dict(l=18, r=18, t=76, b=96),
            xaxis=dict(title="", tickangle=-30, gridcolor=C_GRID, automargin=True),
            yaxis=dict(
                title="Porcentaje (%)", range=[0, 105],
                ticksuffix="%", gridcolor=C_GRID, automargin=True,
            ),
            legend=_clean_legend_bottom(-0.28),
        )
    ))
    _clean_bar_traces(fig)
    return fig


def fig_bivariate_numeric(df_viz, var):
    """
    Boxplot + histograma de densidad para variable numérica vs Diabetes_binary.
    Devuelve una tupla (fig_box, fig_density).
    """
    if df_viz.empty or var not in df_viz.columns:
        return go.Figure(), go.Figure()

    label   = VAR_LABELS.get(var, var)
    dm_mask = df_viz["Diabetes_binary"] == "Prediabetes / Diabetes"
    vals_si = df_viz[dm_mask][var].dropna()
    vals_no = df_viz[~dm_mask][var].dropna()

    # — Boxplot —
    fig_box = go.Figure()
    for name, vals, color in [
        ("Sin diabetes", vals_no, C_BLUE),
        ("Prediabetes / Diabetes", vals_si, C_RED),
    ]:
        if len(vals) == 0:
            continue
        fig_box.add_trace(go.Box(
            y=vals,
            name=name,
            marker_color=color,
            line_color=color,
            fillcolor=color,
            opacity=0.75,
            boxmean=True,
            whiskerwidth=0.5,
            hovertemplate=f"<b>{name}</b><br>{label}: %{{y:.1f}}<extra></extra>",
        ))

    fig_box.update_layout(**_bi_base_layout(
        f"{label} por diabetes",
        "Mediana, dispersión y media",
        extra=dict(
            height=340,
            showlegend=False,
            margin=dict(l=18, r=18, t=76, b=38),
            yaxis=dict(title=label, gridcolor=C_GRID, automargin=True),
            xaxis=dict(title="", gridcolor=C_GRID, automargin=True),
        )
    ))
    _grid_axes(fig_box)

    # — Densidad (histograma normalizado) —
    fig_dens = go.Figure()
    for name, vals, color in [
        ("Sin diabetes", vals_no, C_BLUE),
        ("Prediabetes / Diabetes", vals_si, C_RED),
    ]:
        if len(vals) == 0:
            continue
        fig_dens.add_trace(go.Histogram(
            x=vals,
            name=name,
            marker_color=color,
            opacity=0.65,
            nbinsx=40,
            histnorm="probability density",
            hovertemplate=f"{label}: %{{x:.1f}}<br>Densidad: %{{y:.4f}}<extra>{name}</extra>",
        ))

    if len(vals_no) > 0:
        fig_dens.add_vline(
            x=float(vals_no.mean()), line_dash="dash",
            line_color=C_BLUE, line_width=1.5,
        )
    if len(vals_si) > 0:
        fig_dens.add_vline(
            x=float(vals_si.mean()), line_dash="dash",
            line_color=C_RED, line_width=1.5,
        )

    fig_dens.update_layout(**_bi_base_layout(
        f"Densidad de {label}",
        "Distribución comparada por grupo",
        extra=dict(
            barmode="overlay",
            height=340,
            margin=dict(l=18, r=18, t=76, b=78),
            xaxis=dict(title=label, gridcolor=C_GRID, automargin=True),
            yaxis=dict(title="Densidad", gridcolor=C_GRID, automargin=True),
            legend=_clean_legend_bottom(-0.24),
        )
    ))
    _grid_axes(fig_dens)

    return fig_box, fig_dens


def fig_bivariate_dispatch(df_viz, var):
    """
    Punto de entrada único. Devuelve según tipo de variable:
      - Binaria  → (fig_grouped, None)
      - Ordinal  → (fig_stacked, None)
      - Numérica → (fig_box, fig_density)
    """
    if var in BINARY_VARS:
        return fig_bivariate_binary(df_viz, var), None
    elif var in ORDINAL_VARS_BI:
        return fig_bivariate_ordinal(df_viz, var), None
    elif var in NUMERIC_VARS_BI:
        return fig_bivariate_numeric(df_viz, var)
    else:
        return go.Figure(), None
