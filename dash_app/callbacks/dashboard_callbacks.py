from dash import Input, Output, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

from src.data_loader import get_data
from src.utils.figures import (
    fig_diabetes_distribution,
    fig_age_distribution,
    fig_bmi_boxplot,
    fig_correlation_heatmap,
    fig_factor_ranking,
    fig_bivariate_binary,
    fig_bivariate_ordinal,
    fig_bivariate_numeric,
    BINARY_VARS,
    ORDINAL_VARS_BI,
    NUMERIC_VARS_BI,
    VAR_LABELS,
)
from src.utils.helpers import (
    compute_kpis,
    compute_factor_ranking,
)

_, DF_VIZ = get_data()

C_BLUE = "#1B3A6B"
C_RED = "#C0392B"
C_BLUEI = "#4A7FC1"
C_MUTED = "#7F8C8D"
C_CARD = "#FAFBFC"


def register_dashboard_callbacks(app):

    # ── Gráficos fijos del dashboard ─────────────────────────────────────────
    @app.callback(
        Output("kpi-total",          "children"),
        Output("kpi-total-sub2",     "children"),
        Output("kpi-prevalencia",    "children"),
        Output("kpi-prevalencia-sub","children"),
        Output("kpi-bmi",            "children"),
        Output("kpi-bmi-sub2",       "children"),
        Output("kpi-age",            "children"),
        Output("kpi-age-sub2",       "children"),
        Output("graph-diabetes-dist","figure"),
        Output("graph-age-dist",     "figure"),
        Output("graph-factor-ranking","figure"),
        Output("graph-bmi-boxplot",  "figure"),
        Output("graph-heatmap",      "figure"),
        Input("dashboard-bivariate-variable", "value"),
    )
    def update_fixed_charts(_selected_var):
        dff = DF_VIZ

        kpis   = compute_kpis(dff)
        f_dist = fig_diabetes_distribution(dff)
        f_age  = fig_age_distribution(dff)

        df_rank = compute_factor_ranking(dff)
        f_rank  = fig_factor_ranking(df_rank)
        f_box   = fig_bmi_boxplot(dff)
        f_heat  = fig_correlation_heatmap(dff)

        return (
            kpis["total"],
            kpis["total_sub"],
            kpis["prevalencia"],
            kpis["prevalencia_sub"],
            kpis["bmi_avg"],
            kpis["bmi_sub"],
            kpis["age_label"],
            kpis["age_sub"],
            f_dist,
            f_age,
            f_rank,
            f_box,
            f_heat,
        )

    # ── Análisis bivariado interactivo ───────────────────────────────────────
    @app.callback(
        Output("dashboard-bivariate-main-graph",      "figure"),
        Output("dashboard-bivariate-density-graph",   "figure"),
        Output("dashboard-bivariate-density-wrapper", "style"),
        Output("dashboard-bivariate-type-badge",      "children"),
        Output("dashboard-bivariate-insight",         "children"),
        Input("dashboard-bivariate-variable",         "value"),
    )
    def update_dashboard_bivariate(selected_var):
        empty_fig = _empty_figure()

        if not selected_var or selected_var not in DF_VIZ.columns:
            return empty_fig, empty_fig, {"display": "none"}, html.Div(), html.Div()

        label = VAR_LABELS.get(selected_var, selected_var)
        badge = _build_bivariate_badge(selected_var)
        density_style = {"display": "none"}
        density_fig = empty_fig

        if selected_var in BINARY_VARS:
            main_fig = fig_bivariate_binary(DF_VIZ, selected_var)
            insight = _build_binary_insight(DF_VIZ, selected_var, label)
        elif selected_var in ORDINAL_VARS_BI:
            main_fig = fig_bivariate_ordinal(DF_VIZ, selected_var)
            insight = _build_ordinal_insight(DF_VIZ, selected_var, label)
        elif selected_var in NUMERIC_VARS_BI:
            main_fig, density_fig = fig_bivariate_numeric(DF_VIZ, selected_var)
            density_style = {"display": "block"}
            insight = _build_numeric_insight(DF_VIZ, selected_var, label)
        else:
            main_fig = empty_fig
            insight = html.Div()

        return main_fig, density_fig, density_style, badge, insight


def _empty_figure():
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _build_bivariate_badge(selected_var):
    if selected_var in BINARY_VARS:
        var_type = "Binaria"
        badge_col = C_BLUE
        graph_desc = "Barras agrupadas: conteo de personas por categoría en cada grupo de diabetes."
    elif selected_var in ORDINAL_VARS_BI:
        var_type = "Ordinal"
        badge_col = C_BLUEI
        graph_desc = "Barras apiladas al 100%: composición porcentual a lo largo del orden natural."
    else:
        var_type = "Numérica"
        badge_col = C_RED
        graph_desc = "Boxplot + curva de densidad: distribución y forma por grupo de diabetes."

    return html.Div([
        html.Span(
            f"Tipo: {var_type}",
            style={
                "display": "inline-block",
                "backgroundColor": badge_col,
                "color": "white",
                "padding": "4px 12px",
                "borderRadius": "999px",
                "fontSize": "0.78rem",
                "fontWeight": "700",
                "marginRight": "10px",
            },
        ),
        html.Span(graph_desc, style={"fontSize": "0.84rem", "color": "#5D6D7E"}),
    ])


def _build_binary_insight(df_viz, var, label):
    try:
        cross = pd.crosstab(df_viz[var], df_viz["Diabetes_binary"])
        if "Prediabetes / Diabetes" not in cross.columns:
            return _insight_card(f"Lectura del análisis - {label}", "Sin datos suficientes para el análisis.")

        totals = cross.sum(axis=1)
        pct_dm = (cross["Prediabetes / Diabetes"] / totals * 100).round(1)
        lines = [f"**{cat}:** {pct_dm.loc[cat]:.1f}% con diabetes" for cat in pct_dm.index]
        text = "\n\n".join(lines) if lines else "Sin datos disponibles."
    except Exception:
        text = "Sin datos disponibles."

    return _insight_card(f"Lectura del análisis - {label}", text)


def _build_ordinal_insight(df_viz, var, label):
    try:
        cross = pd.crosstab(df_viz[var], df_viz["Diabetes_binary"])
        if "Prediabetes / Diabetes" in cross.columns and "Sin diabetes" in cross.columns:
            totals = cross.sum(axis=1)
            pct_dm = (cross["Prediabetes / Diabetes"] / totals * 100).round(1)
            max_cat = pct_dm.idxmax()
            min_cat = pct_dm.idxmin()
            text = (
                f"La categoría con mayor prevalencia de diabetes es **{max_cat}** "
                f"({pct_dm[max_cat]:.1f}%), mientras que **{min_cat}** presenta la menor "
                f"({pct_dm[min_cat]:.1f}%)."
            )
        else:
            text = "Sin datos suficientes para el análisis."
    except Exception:
        text = "Sin datos disponibles."

    return _insight_card(f"Lectura del análisis - {label}", text)


def _build_numeric_insight(df_viz, var, label):
    try:
        dm = df_viz["Diabetes_binary"] == "Prediabetes / Diabetes"
        v_si = df_viz[dm][var].dropna()
        v_no = df_viz[~dm][var].dropna()
        diff = v_si.mean() - v_no.mean()
        text = (
            f"Media en personas **con diabetes:** {v_si.mean():.2f} - "
            f"**sin diabetes:** {v_no.mean():.2f}. "
            f"Diferencia de {abs(diff):.2f} unidades "
            f"({'mayor' if diff > 0 else 'menor'} en el grupo con diabetes)."
        )
    except Exception:
        text = "Sin datos disponibles."

    return _insight_card(f"Lectura del análisis - {label}", text)


def _insight_card(title, text):
    return dbc.Card(
        dbc.CardBody([
            html.P(title, style={
                "fontSize": "0.72rem",
                "fontWeight": "700",
                "textTransform": "uppercase",
                "letterSpacing": "0.08em",
                "color": C_MUTED,
                "marginBottom": "8px",
            }),
            dcc.Markdown(text, style={
                "fontSize": "0.90rem",
                "lineHeight": "1.85",
                "color": "#34495E",
                "marginBottom": "0",
            }),
        ], className="px-4 py-3"),
        className="insight-card",
        style={
            "backgroundColor": C_CARD,
        },
    )
