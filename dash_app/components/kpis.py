from dash import html
import dash_bootstrap_components as dbc


def kpi_card(title, value_id, subtext_id=None, accent_class=None):
    value_classes = "kpi-value"
    if accent_class:
        value_classes += f" {accent_class}"

    children = [
        html.Div(title, className="kpi-label"),
        html.Div("—", id=value_id, className=value_classes),
    ]

    if subtext_id:
        children.append(
            html.Div(" ", id=subtext_id, className="kpi-subtext")
        )

    return html.Div(
        className="kpi-card",
        children=children
    )


def create_kpi_row():
    return dbc.Row(
        [
            dbc.Col(
                kpi_card(
                    title="Registros filtrados",
                    value_id="kpi-total",
                    subtext_id="kpi-total-subtext"
                ),
                xl=3,
                md=6,
                className="mb-3"
            ),
            dbc.Col(
                kpi_card(
                    title="Prevalencia de diabetes",
                    value_id="kpi-prevalencia",
                    subtext_id="kpi-prevalencia-subtext"
                ),
                xl=3,
                md=6,
                className="mb-3"
            ),
            dbc.Col(
                kpi_card(
                    title="BMI promedio",
                    value_id="kpi-bmi",
                    subtext_id="kpi-bmi-subtext"
                ),
                xl=3,
                md=6,
                className="mb-3"
            ),
            dbc.Col(
                kpi_card(
                    title="Grupo etario dominante",
                    value_id="kpi-age",
                    subtext_id="kpi-age-subtext"
                ),
                xl=3,
                md=6,
                className="mb-3"
            ),
        ],
        className="g-0"
    )