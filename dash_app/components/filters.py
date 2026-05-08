from dash import html, dcc
import dash_bootstrap_components as dbc
from src.utils.figures import BIVARIATE_OPTIONS


def create_filters():
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Variable a comparar con diabetes", className="form-label",
                               style={"fontSize": "0.75rem", "fontWeight": "600",
                                      "textTransform": "uppercase", "letterSpacing": "0.05em",
                                      "color": "#7F8C8D"}),
                    dcc.Dropdown(
                        id="dashboard-bivariate-variable",
                        options=BIVARIATE_OPTIONS,
                        value="HighBP",
                        clearable=False,
                        placeholder="Selecciona una variable",
                        style={"fontSize": "13px"},
                    ),
                ], md=7),
            ]),
        ], className="py-3 px-4"),
    ], className="filter-card mb-4")
