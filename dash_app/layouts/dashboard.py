from dash import html, dcc
import dash_bootstrap_components as dbc
from dash_app.components.cards import create_kpi_row
from dash_app.components.filters import create_filters

C_NAVY = "#0F2447"
C_BLUE = "#1B3A6B"
C_RED = "#C0392B"
C_BLUEI = "#4A7FC1"
C_BG = "#F4F6F9"
C_TEXT = "#2C3E50"
C_MUTED = "#7F8C8D"
C_CARD = "#FAFBFC"
C_BORDER = "#DDE2E8"


def _graph_card(graph_id, title=None, subtitle=None, height=None):
    graph_style = {"height": height} if height else {}
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.H5(
                            title,
                            style={
                                "fontWeight": "700",
                                "fontSize": "1rem",
                                "color": C_NAVY,
                                "marginBottom": "4px",
                            },
                        ) if title else None,
                        html.P(
                            subtitle,
                            style={
                                "fontSize": "0.84rem",
                                "color": C_MUTED,
                                "marginBottom": "14px",
                            },
                        ) if subtitle else None,
                    ]
                ),
                dcc.Graph(
                    id=graph_id,
                    config={"displayModeBar": False, "responsive": True},
                    style=graph_style,
                ),
            ],
            className="p-3",
        ),
        className="border-0 shadow-sm h-100",
        style={
            "borderRadius": "14px",
            "backgroundColor": C_CARD,
        },
    )


def _section_header(kicker, title, subtitle=None):
    return html.Div(
        [
            html.P(
                kicker,
                style={
                    "fontSize": "0.72rem",
                    "fontWeight": "700",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.08em",
                    "color": C_MUTED,
                    "marginBottom": "6px",
                },
            ),
            html.H4(
                title,
                style={
                    "fontWeight": "800",
                    "color": C_NAVY,
                    "fontSize": "1.35rem",
                    "marginBottom": "6px",
                },
            ),
            html.P(
                subtitle,
                style={
                    "fontSize": "0.9rem",
                    "color": "#5D6D7E",
                    "marginBottom": "0",
                },
            ) if subtitle else None,
        ],
        className="mb-3",
    )


def layout_dashboard():
    return dbc.Container(
        [
            # HEADER
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.P(
                                "BRFSS 2015 · 253.680 individuos",
                                style={
                                    "fontSize": "0.72rem",
                                    "fontWeight": "600",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.08em",
                                    "color": C_MUTED,
                                    "marginBottom": "4px",
                                    "marginTop": "24px",
                                },
                            ),
                            html.H4(
                                "Indicadores de riesgo de diabetes",
                                style={
                                    "fontWeight": "800",
                                    "color": C_NAVY,
                                    "marginBottom": "4px",
                                    "fontSize": "1.9rem",
                                },
                            ),
                            html.P(
                                "Panel analítico para explorar patrones clínicos, conductuales y sociodemográficos asociados a diabetes.",
                                style={
                                    "color": "#95A5A6",
                                    "fontSize": "0.9rem",
                                    "marginBottom": "24px",
                                },
                            ),
                        ]
                    ),
                ],
                className="align-items-start",
            ),

            # BLOQUE 1: VISTA GENERAL FIJA
            _section_header(
                "Vista general del estudio",
                "Panorama inicial",
                "Estos elementos ofrecen una lectura base del fenómeno antes del análisis bivariado.",
            ),
            create_kpi_row(),

            dbc.Row(
                [
                    dbc.Col(
                        _graph_card(
                            "graph-diabetes-dist",
                            title="Distribución general de diabetes",
                            subtitle="Composición de la muestra según presencia o ausencia de diabetes/prediabetes.",
                        ),
                        md=5,
                    ),
                    dbc.Col(
                        _graph_card(
                            "graph-age-dist",
                            title="Prevalencia de diabetes por edad",
                            subtitle="Patrón general por grupos etarios dentro de la población analizada.",
                        ),
                        md=7,
                    ),
                ],
                className="g-3 mb-4",
            ),

            # BLOQUE 2: EXPLORACIÓN INTERACTIVA
            _section_header(
                "Exploración interactiva",
                "Análisis bivariado",
                "Selecciona la variable que quieres comparar con el diagnóstico de diabetes.",
            ),
            create_filters(),
            html.Div(id="dashboard-bivariate-type-badge", className="mb-3"),

            dbc.Row(
                [
                    dbc.Col(
                        _graph_card(
                            "dashboard-bivariate-main-graph",
                            title="Variable seleccionada vs diabetes",
                            subtitle="La visualización se adapta al tipo de variable seleccionada.",
                            height="420px",
                        ),
                        md=8,
                    ),
                    dbc.Col(
                        html.Div(id="dashboard-bivariate-insight"),
                        md=4,
                    ),
                ],
                className="g-3 mb-4",
            ),

            html.Div(
                id="dashboard-bivariate-density-wrapper",
                children=[
                    _graph_card(
                        "dashboard-bivariate-density-graph",
                        title="Curva de densidad por grupo de diabetes",
                        subtitle="Disponible cuando la variable seleccionada es numérica.",
                        height="360px",
                    ),
                ],
                className="mb-4",
                style={"display": "none"},
            ),

            # BLOQUE 3: PROFUNDIZACIÓN ANALÍTICA
            _section_header(
                "Profundización analítica",
                "Relaciones complementarias del análisis",
                "Estos gráficos ayudan a contextualizar la importancia relativa de los factores y sus asociaciones.",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        _graph_card(
                            "graph-factor-ranking",
                            title="Ranking de factores asociados",
                            subtitle="Comparación de relevancia analítica entre variables.",
                        ),
                        md=7,
                    ),
                    dbc.Col(
                        _graph_card(
                            "graph-bmi-boxplot",
                            title="Distribución de BMI por grupo",
                            subtitle="Comparación del índice de masa corporal entre categorías de diabetes.",
                        ),
                        md=5,
                    ),
                ],
                className="g-3 mb-4",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        _graph_card(
                            "graph-heatmap",
                            title="Correlaciones entre indicadores",
                            subtitle="Mapa general de relación entre variables del estudio.",
                            height="520px",
                        ),
                        md=12,
                    ),
                ],
                className="g-3 pb-5",
            ),
        ],
        fluid=True,
        className="px-4",
        style={"backgroundColor": C_BG, "minHeight": "100vh"},
    )
