from dash import html
import dash_bootstrap_components as dbc

# ── Paleta consistente con figures.py ────────────────────────────────────────
C_BLUE  = "#1B3A6B"
C_RED   = "#C0392B"
C_BLUEI = "#4A7FC1"
C_AMBER = "#D4A017"


def kpi_card(card_id, sub_id):
    """Shell card — values injected via callback."""
    return dbc.Card([
        dbc.CardBody([
            html.P("—", id=sub_id,
                   className="kpi-label"),
            html.H3("—", id=card_id,
                    className="kpi-value"),
        ], className="py-3 px-4"),
    ], className="kpi-card h-100")


def create_kpi_row():
    return dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.P("Individuos analizados", id="kpi-total-sub",
                           className="kpi-label"),
                    html.H3("—", id="kpi-total",
                            className="kpi-value"),
                    html.P("—", id="kpi-total-sub2",
                           className="kpi-subtext"),
                ], className="py-3 px-4"),
            ], className="kpi-card h-100", style={"--kpi-accent": C_BLUE}),
            md=3),

        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.P("Prevalencia de diabetes", id="kpi-prev-sub",
                           className="kpi-label"),
                    html.H3("—", id="kpi-prevalencia",
                            className="kpi-value"),
                    html.P("—", id="kpi-prevalencia-sub",
                           className="kpi-subtext"),
                ], className="py-3 px-4"),
            ], className="kpi-card h-100", style={"--kpi-accent": C_RED}),
            md=3),

        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.P("IMC promedio", id="kpi-bmi-sub",
                           className="kpi-label"),
                    html.H3("—", id="kpi-bmi",
                            className="kpi-value"),
                    html.P("—", id="kpi-bmi-sub2",
                           className="kpi-subtext"),
                ], className="py-3 px-4"),
            ], className="kpi-card h-100", style={"--kpi-accent": C_BLUEI}),
            md=3),

        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.P("Grupo etario dominante", id="kpi-age-sub",
                           className="kpi-label"),
                    html.H3("—", id="kpi-age",
                            className="kpi-value"),
                    html.P("—", id="kpi-age-sub2",
                           className="kpi-subtext"),
                ], className="py-3 px-4"),
            ], className="kpi-card h-100", style={"--kpi-accent": C_AMBER}),
            md=3),

    ], className="g-3 mb-4")
