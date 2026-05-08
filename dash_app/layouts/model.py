"""
Layout: Modelo Predictivo (pestaña /predict).

Estructura visual coherente con el resto del dashboard:
    1. Hero / encabezado académico con nota no diagnóstica
    2. Modelo recomendado con KPIs destacados
    3. Selector interactivo de modelos con métricas y descripción dinámica
    4. Simulador de predicción individual
    5. Evaluación visual del mejor modelo (matriz, ROC, PR, importancia)
    6. Comparación general entre modelos (tabla + gráfica)

Reutiliza los estilos CSS ya definidos: `kpi-card`, `graph-card`,
`section-kicker`, `section-title`, `analysis-hero`, `filter-card`.

Clases CSS NUEVAS introducidas (definidas en assets/styles.css):
    - sec-header / sec-header-icon / sec-header-text / sec-header-kicker
      sec-header-title / sec-header-subtitle  →  encabezados de sección
    - sim-group / sim-group-header / sim-group-icon / sim-group-title
      sim-group-hint  →  bloques internos del simulador
    - btn-predict / btn-clear  →  botones del simulador
    - result-card / result-card-banner / result-banner-icon /
      result-banner-kicker / result-banner-title / result-prob-block /
      result-prob-label / result-prob-value / result-prob-bar /
      result-prob-bar-fill / result-prob-scale / result-body / result-tag /
      result-section-title / result-disclaimer  →  panel de resultado
    - cm-mini-grid / cm-mini-card / cm-mini-tag / cm-mini-value /
      cm-mini-desc  →  mini badges TN/FP/FN/TP
    - insight-block / insight-accent / insight-title / insight-list /
      insight-callout / topfeat-item / topfeat-rank / topfeat-name
      →  tarjetas interpretativas con bullets jerarquizados
    - kpi-icon  →  iconito decorativo en cada KPI

IDs nuevos añadidos (gestionados por callbacks):
    - sim-clear-button   →  botón "Limpiar formulario"
"""

from dash import html, dcc
import dash_bootstrap_components as dbc

from src.utils.model_utils import (
    AGE_OPTIONS, BINARY_OPTIONS, EDUCATION_OPTIONS, FEATURE_LABELS_ES,
    GENHLTH_OPTIONS, INCOME_OPTIONS, MODEL_FILES, MODEL_LABELS_ES, SEX_OPTIONS,
    get_best_model_name, get_feature_importance, get_metrics_df,
)

# ── Paleta consistente ───────────────────────────────────────────────────────
C_NAVY  = "#0F2447"
C_BLUE  = "#1B3A6B"
C_RED   = "#C0392B"
C_BLUEI = "#4A7FC1"
C_AQUA  = "#2BB3C8"
C_BG    = "#F4F6F9"
C_TEXT  = "#2C3E50"
C_MUTED = "#7F8C8D"
C_BORDER = "#DDE2E8"
C_CARD  = "#FAFBFC"
C_SOFT  = "#EAF1FB"
C_GREEN = "#1E8449"
C_AMBER = "#C98A16"


# ── Helpers de layout ────────────────────────────────────────────────────────

def _section_header(kicker, title, subtitle=None, icon="fa-solid fa-chart-simple"):
    """Encabezado de sección con icono, kicker, título y subtítulo.

    Usa la clase CSS .sec-header (definida en styles.css) para una
    presentación uniforme y con jerarquía clara entre los grandes bloques
    del módulo de modelado.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.P(kicker, className="sec-header-kicker"),
                    html.H4(title, className="sec-header-title"),
                    html.P(subtitle, className="sec-header-subtitle")
                    if subtitle else None,
                ],
                className="sec-header-text",
            ),
        ],
        className="sec-header",
    )


# Mapa de iconos por métrica (Font Awesome 6 ya está cargado desde app.py)
KPI_ICONS = {
    "Accuracy":  "fa-solid fa-circle-check",
    "Precision": "fa-solid fa-bullseye",
    "Recall":    "fa-solid fa-magnifying-glass",
    "F1-score":  "fa-solid fa-scale-balanced",
    "ROC-AUC":   "fa-solid fa-chart-line",
    "PR-AUC":    "fa-solid fa-crosshairs",
}


def _kpi_card(label, value_id, sub_id, accent=C_BLUE):
    """KPI card reutilizando la clase CSS .kpi-card del dashboard.

    Se añadió un pequeño icono dentro del label (.kpi-icon) coherente con
    el significado de cada métrica. La tipografía y la barra lateral de
    color las gestiona la clase .kpi-card del CSS.
    """
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Span(label),
                    ],
                    className="kpi-label",
                ),
                html.H3("—", id=value_id, className="kpi-value"),
                html.P("—", id=sub_id, className="kpi-subtext"),
            ],
            className="py-3 px-4",
        ),
        className="kpi-card h-100",
        style={"--kpi-accent": accent},
    )


def _graph_card(graph_id, title=None, subtitle=None, height=None, icon=None):
    """Tarjeta contenedora para una gráfica.

    Si se pasa `icon`, se muestra a la izquierda del título para reforzar
    visualmente el tipo de gráfica (matriz, curva, importancia, etc.).
    """
    graph_style = {"height": height} if height else {}

    title_node = None
    if title:
        title_node = html.H5(
            title,
            style={
                "fontWeight": "800", "fontSize": "1.02rem",
                "color": C_NAVY, "marginBottom": "4px",
            },
        )

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        title_node,
                        html.P(
                            subtitle,
                            style={
                                "fontSize": "0.84rem", "color": C_MUTED,
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
        className="graph-card h-100",
        style={"backgroundColor": "#FFFFFF"},
    )


def _interpretation_card(title, body, accent=C_BLUEI):
    """Tarjeta interpretativa simple (texto plano). Mantenida para
    compatibilidad con bloques que aún no migraron a `_insight_block`."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    style={
                        "width": "34px", "height": "4px",
                        "backgroundColor": accent, "borderRadius": "3px",
                        "marginBottom": "12px",
                    }
                ),
                html.H6(
                    title,
                    style={
                        "fontWeight": "800", "fontSize": "0.95rem",
                        "color": C_TEXT, "marginBottom": "10px",
                    },
                ),
                html.P(
                    body,
                    style={
                        "fontSize": "0.88rem", "color": "#5D6D7E",
                        "lineHeight": "1.75", "marginBottom": "0",
                    },
                ),
            ],
            className="p-4",
        ),
        className="border-0 shadow-sm h-100",
        style={"borderRadius": "10px", "backgroundColor": "#FFFFFF"},
    )


def _insight_block(title, items, callout=None, accent=C_BLUEI):
    """Tarjeta interpretativa nueva con bullets jerarquizados.

    `items` es una lista de tuplas (texto_html, tono). El tono puede ser
    "default", "good", "warn" o "bad" y se traduce a la clase del bullet.
    Si se pasa `callout`, se muestra una mini caja resaltada al final.
    """
    bullets = []
    for entry in items:
        if isinstance(entry, tuple):
            content, tone = entry
        else:
            content, tone = entry, "default"
        cls = "bullet" if tone == "default" else f"bullet is-{tone}"
        bullets.append(
            html.Li([html.Span(className=cls), html.Span(content)])
        )

    children = [
        html.Div(className="insight-accent",
                 style={"backgroundColor": accent}),
        html.H6(title, className="insight-title"),
        html.Ul(bullets, className="insight-list"),
    ]
    if callout:
        children.append(html.Div(callout, className="insight-callout"))

    return html.Div(children, className="insight-block")


# ──────────────────────────────────────────────────────────────────────────────
# Bloques principales
# ──────────────────────────────────────────────────────────────────────────────

def _hero():
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    ["Modelado Supervisado"],
                    className="eyebrow-chip",
                ),
                html.H1(
                    "Modelo predictivo de riesgo de prediabetes/diabetes",
                    style={
                        "fontWeight": "800", "color": "#FFFFFF",
                        "fontSize": "2.1rem", "lineHeight": "1.15",
                        "marginTop": "16px", "marginBottom": "12px",
                        "maxWidth": "920px",
                    },
                ),
                html.P(
                    "Esta sección complementa el análisis exploratorio con modelos de "
                    "clasificación supervisada. El objetivo es estimar, a partir de "
                    "variables clínicas, sociodemográficas y de hábitos de salud, si una "
                    "persona podría pertenecer a la clase de prediabetes/diabetes.",
                    style={
                        "color": "rgba(255,255,255,0.82)", "fontSize": "0.98rem",
                        "lineHeight": "1.7", "maxWidth": "880px",
                        "marginBottom": "8px",
                    },
                ),
                html.P(
                    "Los resultados tienen un propósito académico y exploratorio. No "
                    "constituyen un diagnóstico médico ni reemplazan la valoración de "
                    "un profesional de la salud.",
                    style={
                        "color": "rgba(255,255,255,0.62)", "fontSize": "0.86rem",
                        "fontStyle": "italic", "marginBottom": "0",
                        "maxWidth": "880px",
                    },
                ),
            ],
            className="p-4 p-md-5",
        ),
        className="analysis-hero border-0",
        style={"marginTop": "24px", "marginBottom": "32px"},
    )


def _recommended_model_block():
    """Tarjeta destacada con el modelo recomendado y sus KPIs."""
    best = get_best_model_name()
    return dbc.Card(
        dbc.CardBody(
            [
                # Encabezado del bloque
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.P(
                                    "Modelo recomendado",
                                    className="section-kicker",
                                    style={"marginBottom": "4px"},
                                ),
                                html.H3(
                                    best,
                                    id="recommended-model-name",
                                    style={
                                        "fontWeight": "850", "color": C_NAVY,
                                        "fontSize": "1.7rem", "marginBottom": "8px",
                                    },
                                ),
                                html.P(
                                    "Random Forest fue seleccionado como modelo "
                                    "principal porque obtuvo el mejor equilibrio entre "
                                    "precision y recall, evaluado mediante F1-score. "
                                    "Esta métrica es especialmente importante en este "
                                    "problema debido al desbalance de clases.",
                                    style={
                                        "color": "#5D6D7E", "fontSize": "0.92rem",
                                        "lineHeight": "1.7", "marginBottom": "0",
                                        "maxWidth": "780px",
                                    },
                                ),
                            ],
                            md=8,
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.Span(
                                        "Criterio: F1-score",
                                        className="metric-pill",
                                        style={"marginBottom": "8px",
                                               "display": "inline-flex"},
                                    ),
                                    html.Br(),
                                    html.Span(
                                        "Dataset desbalanceado",
                                        className="metric-pill",
                                        style={"display": "inline-flex"},
                                    ),
                                ],
                                style={"textAlign": "right"},
                            ),
                            md=4,
                            className="d-none d-md-block",
                        ),
                    ],
                    className="mb-4 align-items-center",
                ),

                # KPIs del modelo recomendado
                dbc.Row(
                    [
                        dbc.Col(_kpi_card("Accuracy",  "rec-kpi-acc",  "rec-kpi-acc-sub",
                                          accent=C_BLUE), md=4, lg=2),
                        dbc.Col(_kpi_card("Precision", "rec-kpi-prec", "rec-kpi-prec-sub",
                                          accent=C_BLUEI), md=4, lg=2),
                        dbc.Col(_kpi_card("Recall",    "rec-kpi-rec",  "rec-kpi-rec-sub",
                                          accent=C_RED),  md=4, lg=2),
                        dbc.Col(_kpi_card("F1-score",  "rec-kpi-f1",   "rec-kpi-f1-sub",
                                          accent=C_NAVY), md=4, lg=2),
                        dbc.Col(_kpi_card("ROC-AUC",   "rec-kpi-roc",  "rec-kpi-roc-sub",
                                          accent=C_GREEN), md=4, lg=2),
                        dbc.Col(_kpi_card("PR-AUC",    "rec-kpi-pr",   "rec-kpi-pr-sub",
                                          accent=C_AMBER), md=4, lg=2),
                    ],
                    className="g-3 mb-3",
                ),

                # Interpretación
                html.Div(
                    [
                        html.P(
                            [
                                html.I(className="fa-solid fa-lightbulb",
                                       style={"marginRight": "6px",
                                              "color": C_BLUEI}),
                                "¿Cómo leer estas métricas?",
                            ],
                            style={
                                "fontWeight": "700", "color": C_NAVY,
                                "fontSize": "0.86rem", "marginBottom": "6px",
                            },
                        ),
                        html.P(
                            "Aunque el accuracy es relativamente alto, no es la métrica "
                            "más informativa en este caso por el desbalance de clases. "
                            "El recall indica la capacidad del modelo para detectar "
                            "personas con prediabetes/diabetes, mientras que la "
                            "precision muestra qué tan confiables son sus predicciones "
                            "positivas. El F1-score resume el equilibrio entre ambas y "
                            "el ROC-AUC mide la capacidad general de discriminación.",
                            style={
                                "fontSize": "0.86rem", "color": "#5D6D7E",
                                "lineHeight": "1.75", "marginBottom": "0",
                            },
                        ),
                    ],
                    style={
                        "backgroundColor": C_SOFT, "borderLeft": f"3px solid {C_BLUEI}",
                        "padding": "12px 16px", "borderRadius": "8px",
                    },
                ),
            ],
            className="p-4 p-md-4",
        ),
        className="border-0",
        style={
            "borderRadius": "12px",
            "background": "linear-gradient(180deg, #FFFFFF 0%, #FAFCFE 100%)",
            "border": "1px solid #E5EAF2",
            "boxShadow": "0 14px 34px rgba(15, 36, 71, 0.10)",
            "marginBottom": "36px",
        },
    )


def _model_explorer_block():
    """Selector + métricas dinámicas + descripción del modelo seleccionado."""
    model_options = [
        {"label": MODEL_LABELS_ES.get(m, m), "value": m}
        for m in MODEL_FILES.keys()
    ]
    return html.Div(
        [
            _section_header(
                "Comparativo individual",
                "Explora cada modelo entrenado",
                "Selecciona un modelo del menú para ver sus métricas y una "
                "descripción académica de su funcionamiento.",
                icon="fa-solid fa-layer-group",
            ),

            dbc.Card(
                dbc.CardBody(
                    [
                        # Selector
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Modelo a explorar",
                                            className="form-label",
                                            style={
                                                "fontSize": "0.75rem",
                                                "fontWeight": "700",
                                                "textTransform": "uppercase",
                                                "letterSpacing": "0.05em",
                                                "color": "#5D6D7E",
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="model-explorer-select",
                                            options=model_options,
                                            value=get_best_model_name(),
                                            clearable=False,
                                            placeholder="Selecciona un modelo",
                                            style={"fontSize": "13px"},
                                        ),
                                    ],
                                    md=6, lg=5,
                                ),
                            ]
                        ),
                    ],
                    className="py-3 px-4",
                ),
                className="filter-card mb-4",
            ),

            # Descripción del modelo + KPIs dinámicos
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.P(
                                        "Descripción del modelo",
                                        style={
                                            "fontSize": "0.72rem",
                                            "fontWeight": "800",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "0.08em",
                                            "color": C_BLUEI,
                                            "marginBottom": "8px",
                                        },
                                    ),
                                    html.H5(
                                        "—",
                                        id="explorer-model-name",
                                        style={
                                            "fontWeight": "850", "color": C_NAVY,
                                            "fontSize": "1.2rem",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    html.P(
                                        "—",
                                        id="explorer-model-description",
                                        style={
                                            "fontSize": "0.88rem",
                                            "color": "#5D6D7E",
                                            "lineHeight": "1.8",
                                            "marginBottom": "12px",
                                        },
                                    ),
                                    html.Div(
                                        id="explorer-model-interpretation",
                                        style={
                                            "backgroundColor": C_SOFT,
                                            "borderLeft": f"3px solid {C_BLUEI}",
                                            "padding": "10px 14px",
                                            "borderRadius": "6px",
                                            "fontSize": "0.85rem",
                                            "color": "#34495E",
                                            "lineHeight": "1.7",
                                        },
                                    ),
                                ],
                                className="p-4",
                            ),
                            className="border-0 shadow-sm h-100",
                            style={"borderRadius": "10px",
                                   "backgroundColor": "#FFFFFF"},
                        ),
                        md=5,
                    ),
                    dbc.Col(
                        dbc.Row(
                            [
                                dbc.Col(_kpi_card("Accuracy",  "exp-kpi-acc",
                                                  "exp-kpi-acc-sub", accent=C_BLUE),
                                        xs=6, md=4),
                                dbc.Col(_kpi_card("Precision", "exp-kpi-prec",
                                                  "exp-kpi-prec-sub", accent=C_BLUEI),
                                        xs=6, md=4),
                                dbc.Col(_kpi_card("Recall",    "exp-kpi-rec",
                                                  "exp-kpi-rec-sub", accent=C_RED),
                                        xs=6, md=4),
                                dbc.Col(_kpi_card("F1-score",  "exp-kpi-f1",
                                                  "exp-kpi-f1-sub", accent=C_NAVY),
                                        xs=6, md=4),
                                dbc.Col(_kpi_card("ROC-AUC",   "exp-kpi-roc",
                                                  "exp-kpi-roc-sub", accent=C_GREEN),
                                        xs=6, md=4),
                                dbc.Col(_kpi_card("PR-AUC",    "exp-kpi-pr",
                                                  "exp-kpi-pr-sub", accent=C_AMBER),
                                        xs=6, md=4),
                            ],
                            className="g-3 h-100",
                        ),
                        md=7,
                    ),
                ],
                className="g-3 mb-5 align-items-stretch",
            ),
        ]
    )


# ──────────────────────────────────────────────────────────────────────────────
# Simulador interactivo
# ──────────────────────────────────────────────────────────────────────────────

def _sim_group(icon, title, hint, body):
    """Wrapper visual para cada grupo del formulario del simulador.

    Cada grupo (Sociodemográfico, Indicadores clínicos, etc.) se envuelve
    en una mini card propia (.sim-group) con header, icono y contenido,
    para distinguir mejor los conjuntos de inputs.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.H6(title, className="sim-group-title"),
                    html.Span(hint, className="sim-group-hint") if hint else None,
                ],
                className="sim-group-header",
            ),
            html.Div(body),
        ],
        className="sim-group",
    )


def _simulator_form_inputs():
    """Genera los inputs del formulario del simulador, ahora organizados
    en cards internas por categoría (.sim-group)."""
    label_style = {
        "fontSize": "0.74rem", "fontWeight": "700",
        "textTransform": "uppercase", "letterSpacing": "0.04em",
        "color": "#5D6D7E", "marginBottom": "6px",
    }

    def _dropdown(input_id, options, value, label):
        return html.Div(
            [
                html.Label(label, style=label_style),
                dcc.Dropdown(
                    id=input_id, options=options, value=value,
                    clearable=False, style={"fontSize": "13px"},
                ),
            ],
            className="mb-3",
        )

    def _slider(input_id, min_val, max_val, value, label, step=1, marks=None):
        if marks is None:
            marks = {min_val: str(min_val), max_val: str(max_val)}
        return html.Div(
            [
                html.Label(label, style=label_style),
                dcc.Slider(
                    id=input_id, min=min_val, max=max_val, value=value,
                    step=step, marks=marks,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ],
            className="mb-3",
        )

    bmi_marks = {15: "15", 25: "25", 35: "35", 45: "45", 60: "60"}
    days_marks = {0: "0", 10: "10", 20: "20", 30: "30"}

    # ── Bloque 1: Sociodemográfico ─────────────────────────────────────────
    g1 = _sim_group(
        icon="fa-solid fa-user",
        title="Sociodemográfico",
        hint="Perfil básico",
        body=dbc.Row(
            [
                dbc.Col(_dropdown("sim-Sex", SEX_OPTIONS, 0, "Sexo"), md=6),
                dbc.Col(_dropdown("sim-Age", AGE_OPTIONS, 9, "Grupo de edad"), md=6),
                dbc.Col(_dropdown("sim-Education", EDUCATION_OPTIONS, 5,
                                  "Nivel educativo"), md=6),
                dbc.Col(_dropdown("sim-Income", INCOME_OPTIONS, 6,
                                  "Nivel de ingreso"), md=6),
            ],
            className="g-2",
        ),
    )

    # ── Bloque 2: Indicadores clínicos ────────────────────────────────────
    g2 = _sim_group(
        icon="fa-solid fa-heart-pulse",
        title="Indicadores clínicos",
        hint="Antecedentes y BMI",
        body=dbc.Row(
            [
                dbc.Col(_dropdown("sim-HighBP", BINARY_OPTIONS, 0,
                                  "Hipertensión"), md=6),
                dbc.Col(_dropdown("sim-HighChol", BINARY_OPTIONS, 0,
                                  "Colesterol alto"), md=6),
                dbc.Col(_dropdown("sim-CholCheck", BINARY_OPTIONS, 1,
                                  "Chequeo de colesterol (últimos 5 años)"), md=6),
                dbc.Col(_dropdown("sim-Stroke", BINARY_OPTIONS, 0,
                                  "ACV previo"), md=6),
                dbc.Col(_dropdown("sim-HeartDiseaseorAttack", BINARY_OPTIONS, 0,
                                  "Enfermedad cardíaca o infarto"), md=6),
                dbc.Col(_slider("sim-BMI", 12, 60, 27,
                                "Índice de masa corporal (BMI)", step=1,
                                marks=bmi_marks), md=6),
            ],
            className="g-2",
        ),
    )

    # ── Bloque 3: Hábitos y estilo de vida ────────────────────────────────
    g3 = _sim_group(
        icon="fa-solid fa-apple-whole",
        title="Hábitos y estilo de vida",
        hint="Tabaquismo, dieta, ejercicio",
        body=dbc.Row(
            [
                dbc.Col(_dropdown("sim-Smoker", BINARY_OPTIONS, 0,
                                  "Fumador (≥100 cigarrillos en la vida)"), md=6),
                dbc.Col(_dropdown("sim-PhysActivity", BINARY_OPTIONS, 1,
                                  "Actividad física en los últimos 30 días"), md=6),
                dbc.Col(_dropdown("sim-Fruits", BINARY_OPTIONS, 1,
                                  "Consume frutas a diario"), md=6),
                dbc.Col(_dropdown("sim-Veggies", BINARY_OPTIONS, 1,
                                  "Consume verduras a diario"), md=6),
                dbc.Col(_dropdown("sim-HvyAlcoholConsump", BINARY_OPTIONS, 0,
                                  "Consumo elevado de alcohol"), md=6),
            ],
            className="g-2",
        ),
    )

    # ── Bloque 4: Salud percibida y acceso ────────────────────────────────
    g4 = _sim_group(
        icon="fa-solid fa-stethoscope",
        title="Salud percibida y acceso",
        hint="Bienestar y cobertura",
        body=dbc.Row(
            [
                dbc.Col(_dropdown("sim-GenHlth", GENHLTH_OPTIONS, 3,
                                  "Salud general percibida"), md=6),
                dbc.Col(_dropdown("sim-DiffWalk", BINARY_OPTIONS, 0,
                                  "Dificultad para caminar / subir escaleras"), md=6),
                dbc.Col(_dropdown("sim-AnyHealthcare", BINARY_OPTIONS, 1,
                                  "Cobertura médica"), md=6),
                dbc.Col(_dropdown("sim-NoDocbcCost", BINARY_OPTIONS, 0,
                                  "Barrera económica de acceso a salud"), md=6),
                dbc.Col(_slider("sim-MentHlth", 0, 30, 0,
                                "Días con mala salud mental (último mes)",
                                step=1, marks=days_marks), md=6),
                dbc.Col(_slider("sim-PhysHlth", 0, 30, 0,
                                "Días con mala salud física (último mes)",
                                step=1, marks=days_marks), md=6),
            ],
            className="g-2",
        ),
    )

    return [g1, g2, g3, g4]


def _simulator_block():
    return html.Div(
        [
            _section_header(
                "Simulador interactivo",
                "Simulador de predicción individual",
                "Ingrese las características de una persona para estimar si el "
                "modelo la clasifica dentro del grupo de prediabetes/diabetes. "
                "Esta simulación tiene fines académicos y no reemplaza una "
                "valoración médica.",
                icon="fa-solid fa-flask-vial",
            ),

            dbc.Row(
                [
                    # Formulario
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div(_simulator_form_inputs()),

                                    # Bloque 5: Parámetros del modelo + acciones
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H6(
                                                        "Parámetros del modelo",
                                                        className="sim-group-title",
                                                    ),
                                                    html.Span(
                                                        "Selecciona el modelo y ejecuta la predicción",
                                                        className="sim-group-hint",
                                                    ),
                                                ],
                                                className="sim-group-header",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        html.Div(
                                                            [
                                                                html.Label(
                                                                    "Modelo para la simulación",
                                                                    style={
                                                                        "fontSize": "0.74rem",
                                                                        "fontWeight": "700",
                                                                        "textTransform": "uppercase",
                                                                        "letterSpacing": "0.04em",
                                                                        "color": "#5D6D7E",
                                                                        "marginBottom": "6px",
                                                                    },
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="sim-model-select",
                                                                    options=[
                                                                        {"label": MODEL_LABELS_ES.get(m, m),
                                                                         "value": m}
                                                                        for m in MODEL_FILES.keys()
                                                                    ],
                                                                    value=get_best_model_name(),
                                                                    clearable=False,
                                                                    style={"fontSize": "13px"},
                                                                ),
                                                            ]
                                                        ),
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        html.Div(
                                                            [
                                                                dbc.Button(
                                                                    "Predecir riesgo",
                                                                    id="sim-predict-button",
                                                                    n_clicks=0,
                                                                    className="btn-predict",
                                                                    style={
                                                                        "flex": "1 1 60%",
                                                                        "marginRight": "8px",
                                                                    },
                                                                ),
                                                                dbc.Button(
                                                                    "Limpiar",
                                                                    id="sim-clear-button",
                                                                    n_clicks=0,
                                                                    className="btn-clear",
                                                                    style={"flex": "0 0 auto"},
                                                                ),
                                                            ],
                                                            style={
                                                                "display": "flex",
                                                                "gap": "8px",
                                                                "marginTop": "20px",
                                                            },
                                                        ),
                                                        md=6,
                                                    ),
                                                ],
                                                className="align-items-end g-3",
                                            ),
                                        ],
                                        className="sim-group",
                                        style={"marginBottom": "18px"},
                                    ),
                                ],
                                className="p-4",
                            ),
                            className="border-0 shadow-sm",
                            style={"borderRadius": "12px",
                                   "backgroundColor": "#FFFFFF"},
                        ),
                        md=12, lg=12,
                    ),

                    # Panel de resultado
                    dbc.Col(
                        html.Div(
                            id="sim-result-panel",
                            children=_default_result_panel(),
                        ),
                        md=12, lg=12,
                    ),
                ],
                className="g-3 mb-5",
            ),
        ]
    )


def _default_result_panel():
    """Panel inicial (antes de hacer clic). El callback lo reemplaza."""
    return html.Div(
        dbc.Card(
            [
                # Banner superior
                html.Div(
                    [
                        html.Div(
                            [
                                html.P("Resultado clínico estimado",
                                       className="result-banner-kicker"),
                                html.H5("Listo para simular",
                                        className="result-banner-title"),
                            ]
                        ),
                    ],
                    className="result-card-banner is-idle",
                ),

                # Cuerpo
                html.Div(
                    [
                        html.P(
                            "Configure las características del perfil en el "
                            "formulario y presione “Predecir riesgo” para "
                            "obtener la clasificación estimada y la "
                            "probabilidad asociada.",
                            style={
                                "fontSize": "0.88rem", "color": "#5D6D7E",
                                "lineHeight": "1.7", "marginTop": "4px",
                                "marginBottom": "10px",
                            },
                        ),

                        html.P("Próximos pasos",
                               className="result-section-title"),
                        html.Ul(
                            [
                                html.Li("Completa los 4 grupos del formulario."),
                                html.Li("Elige el modelo a utilizar."),
                                html.Li("Ejecuta la predicción."),
                            ],
                            style={
                                "fontSize": "0.84rem", "color": "#5D6D7E",
                                "lineHeight": "1.7", "marginBottom": "0",
                                "paddingLeft": "18px",
                            },
                        ),

                        html.Div(
                            [
                                "Esta herramienta no constituye un diagnóstico "
                                "médico. Sus resultados se basan en patrones "
                                "observados en el dataset BRFSS 2015 y deben "
                                "interpretarse con criterio académico.",
                            ],
                            className="result-disclaimer",
                        ),
                    ],
                    className="result-body",
                ),
            ],
            className="result-card",
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Bloque de evaluación
# ──────────────────────────────────────────────────────────────────────────────

def _confusion_insight_block():
    """Panel didáctico que acompaña a la matriz de confusión.

    En lugar de un párrafo plano, se muestran widgets con bullets
    jerarquizados que explican qué significa cada cuadrante, cuál error
    es más delicado en este problema de salud y cómo interpretar el recall.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.P("Lectura clínica de la matriz",
                                   className="sec-header-kicker"),
                            html.H5("¿Qué nos dice cada cuadrante?",
                                    style={"fontWeight": "850",
                                           "color": C_NAVY,
                                           "margin": "0",
                                           "fontSize": "1.05rem"}),
                        ],
                    ),
                ],
                style={"display": "flex", "alignItems": "center",
                       "gap": "12px", "marginBottom": "14px"},
            ),

            _insight_block(
                title="Cuadrantes y errores del modelo",
                items=[
                    (html.Span([
                        html.Strong("TN (Verdaderos Negativos): "),
                        "personas sin diabetes correctamente clasificadas. "
                        "Son aciertos en la clase mayoritaria.",
                    ]), "good"),
                    (html.Span([
                        html.Strong("TP (Verdaderos Positivos): "),
                        "casos con prediabetes/diabetes detectados por el "
                        "modelo. Son los aciertos clínicamente más valiosos.",
                    ]), "good"),
                    (html.Span([
                        html.Strong("FP (Falsos Positivos): "),
                        "personas sanas marcadas como positivas. Generan "
                        "alertas innecesarias, pero son menos graves que un FN.",
                    ]), "warn"),
                    (html.Span([
                        html.Strong("FN (Falsos Negativos): "),
                        "casos reales que el modelo no detecta. ",
                        html.Em("Son el error más delicado en salud."),
                    ]), "bad"),
                ],
                callout=html.Span([
                    html.Strong("¿Por qué importa el recall? "),
                    "En este problema, perder un caso positivo (FN) puede "
                    "significar dejar a una persona sin atención. Por eso "
                    "priorizamos modelos con recall alto, aun a costa de "
                    "algunos FP.",
                ]),
                accent=C_BLUE,
            ),
        ]
    )


def _top_feature_items(limit=5):
    """Lista visual sincronizada con el CSV usado por la gráfica."""
    df = get_feature_importance()
    if df.empty:
        return [
            html.Div(
                "No hay importancias disponibles.",
                className="topfeat-item topfeat-empty",
            )
        ]

    df = df.sort_values("importance", ascending=False).head(limit)
    items = []
    for _, row in df.iterrows():
        feature = row.get("feature", "")
        label = FEATURE_LABELS_ES.get(feature, feature)
        importance = row.get("importance", None)
        if importance is None:
            value_text = ""
        else:
            value_text = f"{float(importance):.3f}"

        items.append(
            html.Div(
                [
                    html.Span(label, className="topfeat-name"),
                    html.Span(value_text, className="topfeat-value"),
                ],
                className="topfeat-item",
            )
        )
    return items


def _feature_importance_insight_block():
    """Panel mejorado que acompaña a la gráfica de feature importance,
    conectando las variables más relevantes con los hallazgos del EDA."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.P("Variables clave",
                                   className="sec-header-kicker"),
                            html.H5("¿Qué influye más en la predicción?",
                                    style={"fontWeight": "850",
                                           "color": C_NAVY,
                                           "margin": "0",
                                           "fontSize": "1.05rem"}),
                        ],
                    ),
                ],
                style={"display": "flex", "alignItems": "center",
                       "gap": "12px", "marginBottom": "14px"},
            ),

            html.Div(
                [
                    html.P("Top variables que más aportan",
                           className="result-section-title",
                           style={"marginTop": "0"}),

                    html.Div(
                        _top_feature_items(5),
                        style={"marginBottom": "14px"},
                    ),

                    _insight_block(
                        title="Conexión con el análisis exploratorio",
                        items=[
                            (html.Span([
                                "El ", html.Strong("BMI"),
                                " y la ", html.Strong("edad"),
                                " ya mostraban relación clara con la prevalencia "
                                "en el EDA univariado.",
                            ]), "default"),
                            (html.Span([
                                "La ", html.Strong("hipertensión"),
                                " y el ", html.Strong("colesterol alto"),
                                " son comorbilidades frecuentes en los grupos "
                                "positivos.",
                            ]), "default"),
                            (html.Span([
                                "La ", html.Strong("salud general percibida"),
                                " resume múltiples dimensiones del bienestar, "
                                "lo que la convierte en una variable muy "
                                "informativa.",
                            ]), "default"),
                        ],
                        accent=C_GREEN,
                    ),
                ]
            ),
        ]
    )


def _evaluation_block():
    return html.Div(
        [
            _section_header(
                "Evaluación visual",
                "Evaluación del modelo recomendado",
                "Estos diagnósticos visuales ayudan a entender cómo se comporta "
                "el modelo seleccionado: dónde acierta, dónde falla y qué "
                "variables pesan más en sus decisiones.",
                icon="fa-solid fa-chart-pie",
            ),

            # Matriz de confusión + mini badges + interpretación
            dbc.Row(
                [
                    dbc.Col(
                        [
                            _graph_card(
                                "ml-confusion-matrix",
                                title="Matriz de confusión",
                                subtitle="Aciertos y errores del modelo recomendado en el conjunto de prueba.",
                                height="380px",
                                icon="fa-solid fa-table-cells",
                            ),
                        ],
                        md=7,
                    ),
                    dbc.Col(
                        _confusion_insight_block(),
                        md=5,
                    ),
                ],
                className="g-3 mb-5 align-items-stretch",
            ),

            # Curva ROC + PR (encabezado y subtítulo unificado para esta sub-sección)
            html.Div(
                [
                    html.P(
                        "Curvas de desempeño",
                        style={
                            "fontSize": "0.74rem",
                            "fontWeight": "800",
                            "letterSpacing": "0.08em",
                            "color": C_BLUEI,
                            "textTransform": "uppercase",
                            "marginBottom": "4px",
                        },
                    ),
                    html.H5(
                        "ROC y Precision-Recall — modelo recomendado",
                        style={
                            "fontWeight": "850", "color": C_NAVY,
                            "fontSize": "1.05rem", "marginBottom": "4px",
                        },
                    ),
                    html.P(
                        "Las curvas se presentan lado a lado para facilitar la "
                        "comparación. Cada una responde a una pregunta distinta "
                        "y juntas dan una imagen más completa del modelo.",
                        style={"fontSize": "0.86rem", "color": C_MUTED,
                               "marginBottom": "14px"},
                    ),
                ]
            ),

            dbc.Row(
                [
                    dbc.Col(
                        _graph_card(
                            "ml-roc-curve",
                            title="Curva ROC",
                            subtitle="Capacidad de discriminar entre clases en distintos umbrales.",
                            height="380px",
                            icon="fa-solid fa-chart-line",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        _graph_card(
                            "ml-pr-curve",
                            title="Curva Precision-Recall",
                            subtitle="Especialmente útil cuando hay desbalance de clases.",
                            height="380px",
                            icon="fa-solid fa-chart-area",
                        ),
                        md=6,
                    ),
                ],
                className="g-3 mb-3",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        _insight_block(
                            title="Cómo leer la curva ROC",
                            items=[
                                (html.Span([
                                    html.Strong("ROC-AUC "),
                                    "mide la capacidad global del modelo para "
                                    "distinguir entre las dos clases.",
                                ]), "default"),
                                (html.Span([
                                    "Cuanto más cerca de la ", html.Strong("esquina superior izquierda"),
                                    ", mejor el modelo.",
                                ]), "good"),
                                (html.Span([
                                    "La diagonal punteada representa un clasificador "
                                    "aleatorio (AUC = 0.5).",
                                ]), "warn"),
                            ],
                            callout=html.Span([
                                "Aquí se incluyen ", html.Strong("todos los modelos"),
                                " para facilitar la comparación visual.",
                            ]),
                            accent=C_BLUE,
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        _insight_block(
                            title="Cómo leer la curva Precision-Recall",
                            items=[
                                (html.Span([
                                    html.Strong("PR-AUC "),
                                    "es más informativa que ROC cuando la "
                                    "clase positiva es minoritaria.",
                                ]), "default"),
                                (html.Span([
                                    "Permite ver qué tan ", html.Strong("precisas"),
                                    " son las predicciones positivas a distintos "
                                    "niveles de ", html.Strong("recall"), ".",
                                ]), "good"),
                                (html.Span([
                                    "Una curva alta y plana indica que el modelo "
                                    "mantiene buena precisión incluso al recuperar "
                                    "más positivos.",
                                ]), "good"),
                            ],
                            callout=html.Span([
                                "En este dataset desbalanceado, ",
                                html.Strong("PR-AUC es la métrica más fiable"),
                                " para evaluar la clase positiva.",
                            ]),
                            accent=C_RED,
                        ),
                        md=6,
                    ),
                ],
                className="g-3 mb-5 align-items-stretch",
            ),

            # Importancia de variables
            dbc.Row(
                [
                    dbc.Col(
                        _graph_card(
                            "ml-feature-importance",
                            title="Importancia de variables",
                            subtitle="Top variables que más aportan a la predicción del modelo recomendado.",
                            height="500px",
                            icon="fa-solid fa-ranking-star",
                        ),
                        md=7,
                    ),
                    dbc.Col(
                        _feature_importance_insight_block(),
                        md=5,
                    ),
                ],
                className="g-3 mb-5 align-items-stretch",
            ),
        ]
    )


def _comparison_block():
    return html.Div(
        [
            _section_header(
                "Visión global",
                "Comparación general de modelos",
                "Tabla y gráfica comparativas de todos los modelos entrenados, "
                "junto con una interpretación académica del desempeño relativo.",
                icon="fa-solid fa-arrows-left-right-to-line",
            ),

            # Tabla comparativa
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5(
                            "Tabla comparativa de métricas",
                            style={
                                "fontWeight": "800", "fontSize": "1.02rem",
                                "color": C_NAVY, "marginBottom": "4px",
                            },
                        ),
                        html.P(
                            "Modelos ordenados de mayor a menor F1-score. La fila "
                            "destacada corresponde al modelo recomendado.",
                            style={
                                "fontSize": "0.84rem", "color": C_MUTED,
                                "marginBottom": "14px",
                            },
                        ),
                        html.Div(id="ml-models-table"),
                    ],
                    className="p-4",
                ),
                className="border-0 shadow-sm mb-4",
                style={"borderRadius": "10px", "backgroundColor": "#FFFFFF"},
            ),

            # Gráfica de comparación
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5(
                            "Comparación gráfica de métricas",
                            style={
                                "fontWeight": "800", "fontSize": "1.02rem",
                                "color": C_NAVY, "marginBottom": "4px",
                            },
                        ),
                        html.P(
                            "Selecciona las métricas que deseas comparar entre modelos.",
                            style={
                                "fontSize": "0.84rem", "color": C_MUTED,
                                "marginBottom": "14px",
                            },
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Checklist(
                                        id="ml-comp-metrics",
                                        options=[
                                            {"label": " Precision",  "value": "precision"},
                                            {"label": " Recall",     "value": "recall"},
                                            {"label": " F1-score",   "value": "f1"},
                                            {"label": " ROC-AUC",    "value": "roc_auc"},
                                            {"label": " Accuracy",   "value": "accuracy"},
                                            {"label": " PR-AUC",     "value": "pr_auc"},
                                        ],
                                        value=["precision", "recall", "f1", "roc_auc"],
                                        inline=True,
                                        labelStyle={
                                            "marginRight": "18px",
                                            "fontSize": "0.88rem",
                                            "color": "#34495E",
                                            "fontWeight": "600",
                                        },
                                        inputStyle={"marginRight": "6px"},
                                    ),
                                    md=12,
                                ),
                            ],
                            className="mb-3",
                        ),
                        dcc.Graph(
                            id="ml-comparison-graph",
                            config={"displayModeBar": False, "responsive": True},
                            style={"height": "440px"},
                        ),
                    ],
                    className="p-4",
                ),
                className="border-0 shadow-sm mb-4",
                style={"borderRadius": "10px", "backgroundColor": "#FFFFFF"},
            ),

            # Interpretación final
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            style={
                                "width": "44px", "height": "4px",
                                "backgroundColor": C_NAVY, "borderRadius": "3px",
                                "marginBottom": "14px",
                            }
                        ),
                        html.H5(
                            "Lectura final del desempeño",
                            style={
                                "fontWeight": "850", "color": C_NAVY,
                                "fontSize": "1.05rem", "marginBottom": "10px",
                            },
                        ),
                        dcc.Markdown(
                            "Aunque algunos modelos pueden presentar mayor "
                            "**accuracy**, esta métrica no es suficiente para evaluar "
                            "el desempeño en un dataset desbalanceado. En este caso, "
                            "**Random Forest** se selecciona como modelo recomendado "
                            "porque presenta el mejor equilibrio entre precision y "
                            "recall según el F1-score. **XGBoost** y la **Regresión "
                            "Logística** alcanzan valores altos de recall, lo que los "
                            "hace útiles cuando el objetivo prioritario es no perder "
                            "casos positivos, aunque con menor precision. **KNN**, "
                            "pese a tener accuracy elevado, presenta un recall muy "
                            "bajo y por lo tanto no es adecuado para identificar "
                            "correctamente la clase positiva en este problema.",
                            style={
                                "fontSize": "0.92rem", "color": "#34495E",
                                "lineHeight": "1.85", "marginBottom": "0",
                            },
                        ),
                    ],
                    className="p-4",
                ),
                className="border-0 shadow-sm",
                style={"borderRadius": "10px", "backgroundColor": "#FFFFFF"},
            ),
        ]
    )


# ──────────────────────────────────────────────────────────────────────────────
# Layout principal
# ──────────────────────────────────────────────────────────────────────────────

def layout_model():
    return dbc.Container(
        [
            _hero(),
            _recommended_model_block(),
            _model_explorer_block(),
            _simulator_block(),
            _evaluation_block(),
            _comparison_block(),
            html.Div(style={"height": "40px"}),  # padding inferior
        ],
        fluid=True,
        className="px-4 dashboard-shell model-page-shell",
        style={"minHeight": "100vh"},
    )
