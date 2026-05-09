"""
Layout: Modelo Predictivo (pestaña /predict).

────────────────────────────────────────────────────────────────────────────────
NUEVA ESTRUCTURA — Pestaña dividida en DOS SUBPESTAÑAS
────────────────────────────────────────────────────────────────────────────────

La pestaña de Predicción ahora se organiza en dos subpestañas que separan
claramente la parte explicativa de la parte interactiva:

1. SUBPESTAÑA "Información del modelo":
   - Objetivo del modelo y problema que resuelve
   - Justificación del modelo final escogido
   - Métricas de evaluación + interpretación
   - Comparación con otros modelos entrenados
   - Visualizaciones: matriz de confusión, curvas ROC, Precision-Recall,
     importancia de variables, distribución de predicciones, trade-off umbral
   - Conclusiones y limitaciones del modelo

2. SUBPESTAÑA "Realizar predicción":
   - Formulario limpio organizado por grupos (sociodemográfico, clínico,
     hábitos, salud percibida)
   - Validaciones básicas en cada campo
   - Botón para ejecutar la predicción y botón de limpiar
   - Resultado visual con probabilidad, nivel de riesgo e interpretación
   - Mensaje aclaratorio (predicción orientativa, no diagnóstica)

Reutiliza los estilos CSS ya definidos: `kpi-card`, `graph-card`,
`section-kicker`, `section-title`, `analysis-hero`, `filter-card`,
`sec-header`, `sim-group`, `result-card`, `cm-mini-card`, `insight-block`,
`topfeat-item`.

CLASES CSS NUEVAS (definidas en assets/styles.css en este update):
    - prediction-tabs / prediction-tab-label / prediction-tab-label-active /
      prediction-tab-content  →  contenedor de las dos subpestañas
    - info-card / info-card-icon / info-card-title / info-card-text  →
      tarjetas informativas para "objetivo", "problema", "justificación"
    - sim-info-card / sim-info-card-* / sim-info-stat-row  →  tarjeta lateral
      del simulador con info del modelo
    - prediction-disclaimer-banner  →  banner aclaratorio del simulador
    - sim-action-buttons-bottom  →  contenedor de los botones del simulador
"""

from dash import html, dcc
import dash_bootstrap_components as dbc

from src.utils.model_utils import (
    AGE_OPTIONS, BINARY_OPTIONS, EDUCATION_OPTIONS, FEATURE_LABELS_ES,
    GENHLTH_OPTIONS, INCOME_OPTIONS, MODEL_FILES, MODEL_LABELS_ES, SEX_OPTIONS,
    get_best_model_name, get_feature_importance, get_metrics_df,
    get_roc_pr_model_options,
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
    """Encabezado de sección con kicker, título y subtítulo."""
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


# Mapa de iconos por métrica (Font Awesome 6)
KPI_ICONS = {
    "Accuracy":  "fa-solid fa-circle-check",
    "Precision": "fa-solid fa-bullseye",
    "Recall":    "fa-solid fa-magnifying-glass",
    "F1-score":  "fa-solid fa-scale-balanced",
    "ROC-AUC":   "fa-solid fa-chart-line",
    "PR-AUC":    "fa-solid fa-crosshairs",
}


def _kpi_card(label, value_id, sub_id, accent=C_BLUE, extra_class=""):
    """KPI card reutilizable."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div([html.Span(label)], className="kpi-label"),
                html.H3("—", id=value_id, className="kpi-value"),
                html.P("—", id=sub_id, className="kpi-subtext"),
            ],
            className="py-3 px-4",
        ),
        className=f"kpi-card h-100 {extra_class}".strip(),
        style={"--kpi-accent": accent},
    )


def _graph_card(graph_id, title=None, subtitle=None, height=None, icon=None):
    """Tarjeta contenedora para una gráfica."""
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


def _insight_block(title, items, callout=None, accent=C_BLUEI):
    """Tarjeta interpretativa con bullets jerarquizados.

    `items` es una lista de tuplas (texto_html, tono). El tono puede ser
    "default", "good", "warn" o "bad".
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


def _info_card(icon, title, body, accent=C_BLUEI, tag=None):
    """Tarjeta informativa con icono — usada en la sección "Sobre el modelo"."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Span(tag or "Eje metodológico", className="info-card-badge"),
                        html.Span(className="info-card-dot"),
                    ],
                    className="info-card-topline",
                ),
                html.H6(title, className="info-card-title"),
                html.Div(body, className="info-card-text"),
            ],
            className="p-4",
        ),
        className="info-card h-100",
        style={
            "--info-accent": accent,
            "borderRadius": "12px",
            "backgroundColor": "#FFFFFF",
            "border": "1px solid #E5EAF2",
            "boxShadow": "0 4px 14px rgba(15,36,71,0.06)",
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Bloque hero (encabezado general — visible para ambas subpestañas)
# ──────────────────────────────────────────────────────────────────────────────

def _hero():
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(["Modelado Supervisado"], className="eyebrow-chip"),
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
        style={"marginTop": "24px", "marginBottom": "24px"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# SUBPESTAÑA 1 — INFORMACIÓN DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════

def _about_model_block():
    """Bloque introductorio — objetivo, problema, justificación."""
    return html.Div(
        [
            _section_header(
                "Sobre el modelo",
                "¿Qué hace este modelo y por qué?",
                "Resumen del objetivo, el problema que resuelve y los criterios "
                "que llevaron a seleccionar el modelo final.",
                icon="fa-solid fa-circle-info",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        _info_card(
                            icon="fa-solid fa-bullseye",
                            title="Objetivo del modelo",
                            tag="Objetivo",
                            body=html.P(
                                "Estimar la probabilidad de que una persona pertenezca al "
                                "grupo con prediabetes/diabetes a partir de 21 variables "
                                "clínicas, sociodemográficas y de estilo de vida del "
                                "dataset BRFSS 2015. El propósito es servir como "
                                "herramienta exploratoria para apoyar la identificación "
                                "temprana de perfiles de riesgo.",
                                style={"margin": 0},
                            ),
                            accent=C_BLUEI,
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        _info_card(
                            icon="fa-solid fa-stethoscope",
                            title="Problema que resuelve",
                            tag="Desbalance",
                            body=html.P(
                                "La diabetes es una de las enfermedades crónicas más "
                                "prevalentes, y muchos casos se detectan tarde. El modelo "
                                "aborda un problema de clasificación binaria sobre un "
                                "dataset desbalanceado (solo el 13.9% de los registros "
                                "corresponde a la clase positiva), priorizando la "
                                "detección de casos positivos sobre la precisión global.",
                                style={"margin": 0},
                            ),
                            accent=C_RED,
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        _info_card(
                            icon="fa-solid fa-trophy",
                            title="¿Por qué este modelo?",
                            tag="Selección",
                            body=html.P(
                                "Se eligió XGBoost + class_weight porque obtuvo el mayor "
                                "Recall (79.20%) en el conjunto de prueba bajo "
                                "optimización con Optuna y validación cruzada estratificada. "
                                "Es el modelo que mejor detecta personas con prediabetes/"
                                "diabetes, criterio prioritario en este problema de salud "
                                "pública para minimizar falsos negativos.",
                                style={"margin": 0},
                            ),
                            accent=C_GREEN,
                        ),
                        md=4,
                    ),
                ],
                className="g-3 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        _graph_card(
                            "ml-class-imbalance",
                            title="Distribución de la variable objetivo",
                            subtitle="Clase 0: sin diabetes (86.1%) frente a clase 1: prediabetes/diabetes (13.9%).",
                            height="310px",
                        ),
                        md=8,
                    ),
                    dbc.Col(
                        _insight_block(
                            title="Lectura del desbalance",
                            items=[
                                (html.Span([
                                    html.Strong("Clase 0: "),
                                    "218,334 registros sin diabetes (86.1%).",
                                ]), "good"),
                                (html.Span([
                                    html.Strong("Clase 1: "),
                                    "35,346 registros con prediabetes/diabetes (13.9%).",
                                ]), "warn"),
                                (html.Span([
                                    "La razón aproximada es ",
                                    html.Strong("6.2:1"),
                                    ", por eso accuracy no es suficiente como criterio principal.",
                                ]), "default"),
                            ],
                            callout=html.Span([
                                "El proyecto prioriza ", html.Strong("Recall"),
                                " para detectar la mayor cantidad posible de casos positivos.",
                            ]),
                            accent=C_AQUA,
                        ),
                        md=4,
                    ),
                ],
                className="g-3 mb-4 align-items-stretch",
            ),
        ]
    )


def _recommended_model_block():
    """Tarjeta destacada con el modelo recomendado y sus KPIs."""
    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.P(
                                    "Modelo recomendado",
                                    className="section-kicker",
                                    style={"marginBottom": "4px"},
                                ),

                                html.H2(
                                    "—",
                                    id="recommended-model-name",
                                    className="recommended-model-title",
                                ),

                                html.Div(
                                    [
                                        html.Span(
                                            "Técnica de balanceo: —",
                                            id="recommended-balance-technique",
                                            className="recommended-meta-chip",
                                        ),
                                        html.Span(
                                            "Objetivo: maximizar detección",
                                            className="recommended-meta-chip recommended-meta-chip-soft",
                                        ),
                                    ],
                                    className="recommended-meta-row",
                                ),

                                html.P(
                                    "—",
                                    id="recommended-model-context",
                                    className="recommended-model-context",
                                ),
                            ],
                            md=8,
                        ),

                        dbc.Col(
                            html.Div(
                                [
                                    html.Span(
                                        "Criterio principal: Recall",
                                        id="recommended-criterion",
                                        className="metric-pill metric-pill-recall",
                                        style={"marginBottom": "8px", "display": "inline-flex"},
                                    ),
                                    html.Br(),
                                    html.Span(
                                        "Dataset desbalanceado",
                                        id="recommended-dataset-note",
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

                # KPIs del modelo recomendado (6 métricas)
                dbc.Row(
                    [
                        dbc.Col(_kpi_card("Accuracy", "rec-kpi-acc", "rec-kpi-acc-sub",
                                          accent=C_BLUE), md=4, lg=2),
                        dbc.Col(_kpi_card("Precision", "rec-kpi-prec", "rec-kpi-prec-sub",
                                          accent=C_BLUEI), md=4, lg=2),
                        dbc.Col(_kpi_card("Recall", "rec-kpi-rec", "rec-kpi-rec-sub",
                                          accent=C_RED, extra_class="kpi-card-featured"),
                                md=4, lg=2),
                        dbc.Col(_kpi_card("F1-score", "rec-kpi-f1", "rec-kpi-f1-sub",
                                          accent=C_NAVY), md=4, lg=2),
                        dbc.Col(_kpi_card("ROC-AUC", "rec-kpi-roc", "rec-kpi-roc-sub",
                                          accent=C_GREEN), md=4, lg=2),
                        dbc.Col(_kpi_card("PR-AUC", "rec-kpi-pr", "rec-kpi-pr-sub",
                                          accent=C_AMBER), md=4, lg=2),
                    ],
                    className="g-3 mb-3",
                ),

                # Caja explicativa
                html.Div(
                    [
                        html.P(
                            [
                                html.I(
                                    className="fa-solid fa-lightbulb",
                                    style={"marginRight": "6px", "color": C_RED},
                                ),
                                "¿Cómo leer estas métricas?",
                            ],
                            style={
                                "fontWeight": "700", "color": C_NAVY,
                                "fontSize": "0.88rem", "marginBottom": "6px",
                            },
                        ),
                        html.P(
                            "En este proyecto se prioriza el recall porque el objetivo principal "
                            "es detectar la mayor cantidad posible de personas con prediabetes/"
                            "diabetes. Por ello, el modelo recomendado no es necesariamente el más "
                            "preciso ni el de mayor accuracy, sino el que mejor identifica los casos "
                            "positivos. Las demás métricas complementan la interpretación del "
                            "desempeño global.",
                            style={
                                "fontSize": "0.86rem", "color": "#5D6D7E",
                                "lineHeight": "1.75", "marginBottom": "0",
                            },
                        ),
                    ],
                    className="recommended-reading-box",
                ),
            ],
            className="p-4 p-md-4",
        ),
        className="border-0 recommended-panel",
        style={
            "borderRadius": "14px",
            "background": "linear-gradient(180deg, #FFFFFF 0%, #FAFCFE 100%)",
            "border": "1px solid #E5EAF2",
            "boxShadow": "0 14px 34px rgba(15, 36, 71, 0.10)",
            "marginBottom": "36px",
        },
    )


def _metrics_explanation_block():
    """Bloque explicativo de cada métrica."""
    return html.Div(
        [
            _section_header(
                "Interpretación de métricas",
                "¿Qué mide cada métrica?",
                "Cada métrica responde a una pregunta distinta sobre el desempeño del modelo. "
                "Aquí explicamos qué significan en el contexto de este problema de salud.",
                icon="fa-solid fa-book-open",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        _insight_block(
                            title="Métricas de clasificación",
                            items=[
                                (html.Span([
                                    html.Strong("Accuracy: "),
                                    "proporción de aciertos sobre el total. "
                                    "En datasets desbalanceados puede ser engañosa.",
                                ]), "warn"),
                                (html.Span([
                                    html.Strong("Precision: "),
                                    "de los casos que el modelo clasifica como positivos, "
                                    "qué porcentaje realmente lo son. Mide la confiabilidad "
                                    "de las predicciones positivas.",
                                ]), "default"),
                                (html.Span([
                                    html.Strong("Recall (sensibilidad): "),
                                    "de todos los casos realmente positivos, qué porcentaje "
                                    "el modelo logra detectar. ",
                                    html.Em("Métrica principal en este proyecto."),
                                ]), "good"),
                                (html.Span([
                                    html.Strong("F1-score: "),
                                    "media armónica entre precision y recall. Útil cuando "
                                    "se busca un equilibrio entre ambas.",
                                ]), "default"),
                            ],
                            accent=C_BLUEI,
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        _insight_block(
                            title="Métricas basadas en probabilidad",
                            items=[
                                (html.Span([
                                    html.Strong("ROC-AUC: "),
                                    "área bajo la curva ROC. Mide la capacidad global "
                                    "del modelo para discriminar entre las dos clases, "
                                    "sin depender del umbral de clasificación.",
                                ]), "default"),
                                (html.Span([
                                    html.Strong("PR-AUC: "),
                                    "área bajo la curva Precision-Recall. Más informativa "
                                    "que ROC cuando la clase positiva es minoritaria.",
                                ]), "default"),
                                (html.Span([
                                    "Un valor de ", html.Strong("0.5"),
                                    " en ROC-AUC equivale a un clasificador aleatorio.",
                                ]), "warn"),
                                (html.Span([
                                    "Cuanto más cercano a ", html.Strong("1.0"),
                                    ", mejor el desempeño del modelo.",
                                ]), "good"),
                            ],
                            accent=C_GREEN,
                        ),
                        md=6,
                    ),
                ],
                className="g-3 mb-4 align-items-stretch",
            ),
        ]
    )


def _model_explorer_block():
    """Selector separado por modelo base y técnica de balanceo."""

    return html.Div(
        [
            _section_header(
                "Comparativo individual",
                "Explora cada modelo entrenado",
                "Selecciona primero el modelo base y luego la técnica de balanceo para analizar su desempeño.",
                icon="fa-solid fa-layer-group",
            ),

            # Selector
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Modelo base",
                                            className="form-label",
                                            style={
                                                "fontSize": "0.75rem",
                                                "fontWeight": "800",
                                                "textTransform": "uppercase",
                                                "letterSpacing": "0.06em",
                                                "color": "#5D6D7E",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="model-base-dropdown",
                                            options=[
                                                {"label": "Regresión Logística", "value": "Regresión Logística"},
                                                {"label": "Random Forest", "value": "Random Forest"},
                                                {"label": "XGBoost", "value": "XGBoost"},
                                                {"label": "KNN", "value": "KNN"},
                                            ],
                                            value="XGBoost",
                                            clearable=False,
                                            placeholder="Selecciona un modelo base",
                                            style={"fontSize": "13px"},
                                        ),
                                        html.Small(
                                            "Elige el algoritmo principal que deseas analizar.",
                                            style={
                                                "display": "block",
                                                "marginTop": "7px",
                                                "fontSize": "0.78rem",
                                                "color": "#7F8C8D",
                                            },
                                        ),
                                    ],
                                    md=5,
                                ),

                                dbc.Col(
                                    [
                                        html.Label(
                                            "Técnica de balanceo",
                                            className="form-label",
                                            style={
                                                "fontSize": "0.75rem",
                                                "fontWeight": "800",
                                                "textTransform": "uppercase",
                                                "letterSpacing": "0.06em",
                                                "color": "#5D6D7E",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        dcc.RadioItems(
                                            id="balance-technique-radio",
                                            options=[],
                                            value=None,
                                            inline=True,
                                            className="balance-radio-group",
                                            inputClassName="balance-radio-input",
                                            labelClassName="balance-radio-label",
                                        ),
                                        html.Small(
                                            "Solo se muestran estrategias compatibles con el modelo seleccionado.",
                                            style={
                                                "display": "block",
                                                "marginTop": "7px",
                                                "fontSize": "0.78rem",
                                                "color": "#7F8C8D",
                                            },
                                        ),
                                    ],
                                    md=7,
                                ),
                            ],
                            className="g-4 align-items-start",
                        ),

                        html.Div(
                            id="selected-combination-summary",
                            className="selection-summary mt-4",
                        ),
                    ],
                    className="py-4 px-4",
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
                                            "fontWeight": "850",
                                            "color": C_NAVY,
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
                            style={
                                "borderRadius": "10px",
                                "backgroundColor": "#FFFFFF",
                            },
                        ),
                        md=5,
                    ),

                    dbc.Col(
                        dbc.Row(
                            [
                                dbc.Col(_kpi_card("Accuracy", "exp-kpi-acc",
                                                  "exp-kpi-acc-sub", accent=C_BLUE),
                                        xs=6, md=4),
                                dbc.Col(_kpi_card("Precision", "exp-kpi-prec",
                                                  "exp-kpi-prec-sub", accent=C_BLUEI),
                                        xs=6, md=4),
                                dbc.Col(_kpi_card("Recall", "exp-kpi-rec",
                                                  "exp-kpi-rec-sub", accent=C_RED,
                                                  extra_class="kpi-card-featured"),
                                        xs=6, md=4),
                                dbc.Col(_kpi_card("F1-score", "exp-kpi-f1",
                                                  "exp-kpi-f1-sub", accent=C_NAVY),
                                        xs=6, md=4),
                                dbc.Col(_kpi_card("ROC-AUC", "exp-kpi-roc",
                                                  "exp-kpi-roc-sub", accent=C_GREEN),
                                        xs=6, md=4),
                                dbc.Col(_kpi_card("PR-AUC", "exp-kpi-pr",
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


def _confusion_insight_block():
    """Panel didáctico que acompaña a la matriz de confusión."""
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
                                           "color": C_NAVY, "margin": "0",
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
    """Panel mejorado que acompaña a la gráfica de feature importance."""
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
                                           "color": C_NAVY, "margin": "0",
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

                    html.Div(_top_feature_items(5),
                             style={"marginBottom": "14px"}),

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
    """Bloque de evaluación visual del modelo recomendado."""
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

            # Matriz de confusión + interpretación
            dbc.Row(
                [
                    dbc.Col(
                        [
                            _graph_card(
                                "ml-confusion-matrix",
                                title="Matriz de confusión",
                                subtitle="Aciertos y errores del modelo recomendado en el conjunto de prueba.",
                                height="450px",
                            ),
                        ],
                        md=7,
                    ),
                    dbc.Col(_confusion_insight_block(), md=5),
                ],
                className="g-3 mb-5 align-items-stretch",
            ),

            # Curvas ROC + PR
            html.Div(
                [
                    html.P("Curvas de desempeño",
                           style={
                               "fontSize": "0.74rem", "fontWeight": "800",
                               "letterSpacing": "0.08em", "color": C_BLUEI,
                               "textTransform": "uppercase", "marginBottom": "4px",
                           }),
                    html.H5("ROC y Precision-Recall — comparación de modelos",
                            style={"fontWeight": "850", "color": C_NAVY,
                                   "fontSize": "1.05rem", "marginBottom": "4px"}),
                    html.P(
                        "Selecciona uno o varios modelos para comparar su desempeño. "
                        "Cada curva corresponde a la mejor técnica de balanceo encontrada para ese modelo base.",
                        style={"fontSize": "0.86rem", "color": C_MUTED,
                               "marginBottom": "14px"},
                    ),
                    dcc.Dropdown(
                        id="ml-curve-model-selector",
                        options=get_roc_pr_model_options(),
                        value=[opt["value"] for opt in get_roc_pr_model_options()],
                        multi=True,
                        placeholder="Selecciona modelos para comparar",
                        className="curve-model-dropdown",
                    ),
                    html.Div(id="ml-curves-note", className="curve-model-note"),
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
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        _graph_card(
                            "ml-pr-curve",
                            title="Curva Precision-Recall",
                            subtitle="Especialmente útil cuando hay desbalance de clases.",
                            height="380px",
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
                                html.Strong("PR-AUC > ROC-AUC"),
                                " no siempre es alcanzable, pero curvas más altas "
                                "indican mejores modelos.",
                            ]),
                            accent=C_GREEN,
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
                        ),
                        md=7,
                    ),
                    dbc.Col(_feature_importance_insight_block(), md=5),
                ],
                className="g-3 mb-5 align-items-stretch",
            ),

            # ── NUEVA: Distribución de probabilidades predichas ──────────────
            _section_header(
                "Distribución de predicciones",
                "Distribución de probabilidades predichas por clase real",
                "Esta gráfica muestra cómo se distribuyen las probabilidades estimadas por el modelo "
                "para los casos reales positivos y negativos. Una buena separación entre las dos "
                "distribuciones indica que el modelo discrimina bien entre clases.",
                icon="fa-solid fa-chart-area",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        _graph_card(
                            "ml-proba-distribution",
                            title="Distribución de probabilidades estimadas",
                            subtitle="Cómo separa el modelo a los casos positivos y negativos según la probabilidad asignada.",
                            height="430px",
                        ),
                        md=7,
                    ),
                    dbc.Col(
                        _insight_block(
                            title="Cómo interpretar esta gráfica",
                            items=[
                                (html.Span([
                                    "El eje X muestra la ", html.Strong("probabilidad estimada"),
                                    " de que un caso sea positivo (0 a 1).",
                                ]), "default"),
                                (html.Span([
                                    "Idealmente, los casos ",
                                    html.Strong("realmente positivos"),
                                    " (rojo) se concentran a la derecha, "
                                    "y los ", html.Strong("realmente negativos"),
                                    " (verde) a la izquierda.",
                                ]), "good"),
                                (html.Span([
                                    "El umbral de clasificación (línea punteada) decide a partir "
                                    "de qué probabilidad un caso se predice como positivo.",
                                ]), "default"),
                                (html.Span([
                                    "Si las distribuciones se superponen mucho, el modelo "
                                    "tendrá más errores de clasificación.",
                                ]), "warn"),
                            ],
                            accent=C_BLUEI,
                        ),
                        md=5,
                    ),
                ],
                className="g-3 mb-5 align-items-stretch",
            ),

            # ── NUEVA: Trade-off Recall/Precision por umbral ─────────────────
            _section_header(
                "Análisis del umbral",
                "Trade-off Precision / Recall según umbral",
                "Al cambiar el umbral de decisión, las métricas precision y recall se mueven en "
                "sentidos opuestos. Esta gráfica permite ver el efecto del umbral final escogido.",
                icon="fa-solid fa-sliders",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        _graph_card(
                            "ml-threshold-tradeoff",
                            title="Métricas según umbral de clasificación",
                            subtitle="Recall, Precision y F1 al variar el umbral de decisión del modelo final.",
                            height="430px",
                        ),
                        md=7,
                    ),
                    dbc.Col(
                        _insight_block(
                            title="¿Por qué ajustar el umbral?",
                            items=[
                                (html.Span([
                                    "Por defecto, los modelos clasifican como positivo cuando "
                                    "la probabilidad ≥ ", html.Strong("0.50"), ".",
                                ]), "default"),
                                (html.Span([
                                    "En este proyecto se eligió un umbral más bajo "
                                    "para ", html.Strong("priorizar el recall"),
                                    " manteniendo precision ≥ 0.25.",
                                ]), "good"),
                                (html.Span([
                                    "Bajar el umbral aumenta el recall pero reduce la precision: "
                                    "más detección, pero también más falsos positivos.",
                                ]), "warn"),
                                (html.Span([
                                    "La línea punteada vertical marca el ",
                                    html.Strong("umbral final"),
                                    " seleccionado en el notebook.",
                                ]), "default"),
                            ],
                            accent=C_AMBER,
                        ),
                        md=5,
                    ),
                ],
                className="g-3 mb-5 align-items-stretch",
            ),
        ]
    )


def _comparison_block():
    """Tabla y gráfica comparativa de todos los modelos entrenados."""
    return html.Div(
        [
            _section_header(
                "Visión global",
                "Comparación general de modelos",
                "Tabla y gráfica comparativas de todos los modelos entrenados, "
                "junto con una interpretación académica del desempeño relativo.",
                icon="fa-solid fa-arrows-left-right-to-line",
            ),

            # Tabla
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5(
                            "Tabla comparativa de métricas",
                            style={"fontWeight": "800", "fontSize": "1.02rem",
                                   "color": C_NAVY, "marginBottom": "4px"},
                        ),
                        html.P(
                            "Modelos ordenados de mayor a menor Recall. La fila "
                            "destacada corresponde al modelo recomendado.",
                            style={"fontSize": "0.84rem", "color": C_MUTED,
                                   "marginBottom": "14px"},
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
                            style={"fontWeight": "800", "fontSize": "1.02rem",
                                   "color": C_NAVY, "marginBottom": "4px"},
                        ),
                        html.P(
                            "Selecciona las métricas que deseas comparar entre modelos.",
                            style={"fontSize": "0.84rem", "color": C_MUTED,
                                   "marginBottom": "14px"},
                        ),
                        dcc.Checklist(
                            id="ml-comp-metrics",
                            options=[
                                {"label": " Precision", "value": "precision"},
                                {"label": " Recall", "value": "recall"},
                                {"label": " F1-score", "value": "f1"},
                                {"label": " ROC-AUC", "value": "roc_auc"},
                                {"label": " Accuracy", "value": "accuracy"},
                                {"label": " PR-AUC", "value": "pr_auc"},
                            ],
                            value=["precision", "recall", "f1", "roc_auc"],
                            inline=True,
                            className="metric-checklist",
                            inputStyle={"marginRight": "7px"},
                            labelStyle={"fontWeight": "750"},
                        ),
                        dcc.Graph(
                            id="ml-comparison-graph",
                            config={"displayModeBar": False, "responsive": True},
                            style={"height": "500px"},
                        ),
                    ],
                    className="p-4",
                ),
                className="border-0 shadow-sm mb-4",
                style={"borderRadius": "10px", "backgroundColor": "#FFFFFF"},
            ),

            # ── NUEVA: Heatmap Recall por modelo y balanceo ──────────────────
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5(
                            "Desempeño por modelo y técnica de balanceo",
                            style={"fontWeight": "800", "fontSize": "1.02rem",
                                   "color": C_NAVY, "marginBottom": "4px"},
                        ),
                        html.P(
                            "Comparación de Recall, F1-score y ROC-AUC para las combinaciones entrenadas.",
                            style={"fontSize": "0.84rem", "color": C_MUTED,
                                   "marginBottom": "14px"},
                        ),
                        dcc.Graph(
                            id="ml-three-metric-heatmaps",
                            config={"displayModeBar": False, "responsive": True},
                            style={"height": "460px"},
                        ),
                    ],
                    className="p-4",
                ),
                className="border-0 shadow-sm mb-4",
                style={"borderRadius": "10px", "backgroundColor": "#FFFFFF"},
            ),
        ]
    )


def _conclusions_and_limitations_block():
    """Bloque final con conclusiones y limitaciones."""
    return html.Div(
        [
            _section_header(
                "Cierre del modelo",
                "Conclusiones y limitaciones",
                "Síntesis de los hallazgos del modelado y advertencias importantes "
                "para interpretar los resultados.",
                icon="fa-solid fa-flag-checkered",
            ),

            # Banner de conclusiones
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
                            "Conclusiones del modelado",
                            style={"fontWeight": "850", "color": C_NAVY,
                                   "fontSize": "1.05rem", "marginBottom": "10px"},
                        ),
                        dcc.Markdown(
                            "El modelado confirma que la evaluación no debe basarse únicamente "
                            "en **accuracy**, debido al desbalance de la variable objetivo. En "
                            "este contexto, la métrica principal es **Recall**, ya que el "
                            "propósito del modelo es reducir la cantidad de casos positivos no "
                            "detectados en la clase prediabetes/diabetes.\n\n"
                            "A partir de la comparación entre modelos y estrategias de balanceo, "
                            "se seleccionó **XGBoost + class_weight** como modelo recomendado, "
                            "al obtener el **mayor Recall en el conjunto de prueba (79.20%)**. "
                            "Esta elección es coherente con un problema de salud pública donde "
                            "los falsos negativos tienen mayor costo operativo y clínico que las "
                            "alertas que posteriormente pueden verificarse con pruebas adicionales.\n\n"
                            "El proceso fue desarrollado con separación estricta entre entrenamiento "
                            "y prueba, pipelines para evitar fuga de información, validación cruzada "
                            "estratificada de 5 folds y optimización con Optuna. Las variables con "
                            "mayor aporte al modelo, como hipertensión, salud general percibida, "
                            "colesterol alto, dificultad para caminar, edad e IMC, son consistentes "
                            "con los patrones observados en el análisis exploratorio.\n\n"
                            "Adicionalmente, el ajuste del umbral de clasificación permite elevar "
                            "la sensibilidad del modelo cuando el objetivo operativo es maximizar "
                            "la detección de casos positivos, manteniendo una precisión mínima "
                            "aceptable para un escenario de tamizaje.",
                            style={"fontSize": "0.92rem", "color": "#34495E",
                                   "lineHeight": "1.85", "marginBottom": "0"},
                        ),
                    ],
                    className="p-4",
                ),
                className="border-0 shadow-sm mb-4",
                style={"borderRadius": "10px", "backgroundColor": "#FFFFFF"},
            ),

            # Limitaciones / advertencias en dos columnas
            dbc.Row(
                [
                    dbc.Col(
                        _insight_block(
                            title="Consideraciones metodológicas",
                            items=[
                                (html.Span([
                                    html.Strong("Uso no diagnóstico. "),
                                    "El resultado debe interpretarse como apoyo exploratorio "
                                    "y no como confirmación clínica individual.",
                                ]), "bad"),
                                (html.Span([
                                    html.Strong("Datos auto-reportados. "),
                                    "BRFSS 2015 se basa en encuesta poblacional, por lo que "
                                    "puede incorporar sesgos de memoria, acceso o respuesta.",
                                ]), "warn"),
                                (html.Span([
                                    html.Strong("Población específica. "),
                                    "Los datos provienen de Estados Unidos (2015). "
                                    "Aplicar el modelo a otras poblaciones requiere validación.",
                                ]), "warn"),
                                (html.Span([
                                    html.Strong("Precisión moderada. "),
                                    "El modelo prioriza recall, lo que implica un mayor "
                                    "número de falsos positivos.",
                                ]), "warn"),
                            ],
                            accent=C_RED,
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        _insight_block(
                            title="Alcance e interpretación",
                            items=[
                                (html.Span([
                                    html.Strong("Apoyo a tamizaje. "),
                                    "Útil para priorizar perfiles que podrían requerir "
                                    "evaluación adicional o seguimiento preventivo.",
                                ]), "good"),
                                (html.Span([
                                    html.Strong("Lectura de factores asociados. "),
                                    "Las variables relevantes son coherentes con el EDA y "
                                    "con factores clínicos esperados.",
                                ]), "good"),
                                (html.Span([
                                    html.Strong("Proceso reproducible. "),
                                    "El flujo conserva separación train/test, validación "
                                    "estratificada, pipelines y artefactos serializados.",
                                ]), "good"),
                                (html.Span([
                                    html.Strong("Criterio transparente. "),
                                    "La selección se fundamenta en Recall, métrica alineada "
                                    "con la detección de la clase positiva.",
                                ]), "good"),
                            ],
                            accent=C_GREEN,
                        ),
                        md=6,
                    ),
                ],
                className="g-3 mb-3 align-items-stretch",
            ),
        ]
    )


def _info_subtab_content():
    """Contenido completo de la subpestaña 'Información del modelo'."""
    return html.Div(
        [
            _about_model_block(),
            _recommended_model_block(),
            _metrics_explanation_block(),
            _model_explorer_block(),
            _evaluation_block(),
            _comparison_block(),
            _conclusions_and_limitations_block(),
        ],
        className="prediction-tab-content",
    )


# ══════════════════════════════════════════════════════════════════════════════
# SUBPESTAÑA 2 — REALIZAR PREDICCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _sim_group(icon, title, hint, body):
    """Wrapper visual para cada grupo del formulario del simulador."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        html.I(className=icon),
                        className="sim-group-icon",
                    ),
                    html.Div(
                        [
                            html.H6(title, className="sim-group-title"),
                            html.Span(hint, className="sim-group-hint")
                            if hint else None,
                        ]
                    ),
                ],
                className="sim-group-header",
            ),
            html.Div(body),
        ],
        className="sim-group",
    )


def _simulator_form_inputs():
    """Genera los inputs del formulario, organizados en 4 grupos."""
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

    # Bloque 1: Sociodemográfico
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

    # Bloque 2: Indicadores clínicos
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

    # Bloque 3: Hábitos y estilo de vida
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

    # Bloque 4: Salud percibida y acceso
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


def _model_info_card():
    """Tarjeta lateral con info del modelo usado en la predicción."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Span("Modelo en uso", className="sim-info-card-kicker"),
                        html.H5("XGBoost + Class weight",
                                className="sim-info-card-title"),
                    ]
                ),

                html.Div(
                    [
                        html.Div(
                            [
                                html.I(className="fa-solid fa-magnifying-glass",
                                       style={"color": C_RED, "marginRight": "6px"}),
                                html.Span("Recall", className="sim-info-stat-label"),
                                html.Span("79.20%", className="sim-info-stat-value"),
                            ],
                            className="sim-info-stat-row",
                        ),
                        html.Div(
                            [
                                html.I(className="fa-solid fa-bullseye",
                                       style={"color": C_BLUEI, "marginRight": "6px"}),
                                html.Span("Precision", className="sim-info-stat-label"),
                                html.Span("30.60%", className="sim-info-stat-value"),
                            ],
                            className="sim-info-stat-row",
                        ),
                        html.Div(
                            [
                                html.I(className="fa-solid fa-chart-line",
                                       style={"color": C_GREEN, "marginRight": "6px"}),
                                html.Span("ROC-AUC", className="sim-info-stat-label"),
                                html.Span("82.71%", className="sim-info-stat-value"),
                            ],
                            className="sim-info-stat-row",
                        ),
                        html.Div(
                            [
                                html.I(className="fa-solid fa-sliders",
                                       style={"color": C_AMBER, "marginRight": "6px"}),
                                html.Span("Umbral final", className="sim-info-stat-label"),
                                html.Span("0.33", className="sim-info-stat-value"),
                            ],
                            className="sim-info-stat-row",
                        ),
                    ],
                    className="sim-info-stats-block",
                ),

                html.P(
                    "Modelo final entrenado y optimizado con Optuna sobre el "
                    "dataset BRFSS 2015. Prioriza la detección (recall) de "
                    "personas con prediabetes/diabetes.",
                    className="sim-info-card-body",
                ),

                html.Div(
                    [
                        html.I(className="fa-solid fa-circle-exclamation",
                               style={"marginRight": "8px", "color": C_AMBER}),
                        html.Span(
                            "Esta predicción es orientativa y no reemplaza una "
                            "valoración médica profesional.",
                        ),
                    ],
                    className="sim-info-card-disclaimer",
                ),
            ],
            className="p-4",
        ),
        className="sim-info-card",
    )


def _default_result_panel():
    """Panel inicial (antes de hacer clic). El callback lo reemplaza."""
    return html.Div(
        dbc.Card(
            [
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

                html.Div(
                    [
                        html.P(
                            "Configure las características del perfil en el "
                            "formulario y presione \"Predecir riesgo\" para "
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
                                html.Li("Revisa los valores del perfil."),
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


def _simulator_block():
    """Bloque del simulador interactivo (subpestaña 'Realizar predicción')."""
    return html.Div(
        [
            _section_header(
                "Simulador interactivo",
                "Realizar predicción de riesgo",
                "Ingrese las características de una persona para estimar si el "
                "modelo la clasifica dentro del grupo de prediabetes/diabetes. "
                "Esta simulación tiene fines académicos y NO reemplaza una "
                "valoración médica.",
                icon="fa-solid fa-flask-vial",
            ),

            # Aviso destacado al inicio
            html.Div(
                [
                    html.I(className="fa-solid fa-circle-info",
                           style={"marginRight": "10px", "color": C_BLUEI,
                                  "fontSize": "1.1rem"}),
                    html.Span([
                        html.Strong("Importante: "),
                        "Esta es una herramienta orientativa de tamizaje. La predicción "
                        "se basa en patrones del dataset BRFSS 2015 (USA, 2015) y NO "
                        "constituye un diagnóstico médico. Consulta siempre con un "
                        "profesional de la salud.",
                    ]),
                ],
                className="prediction-disclaimer-banner",
            ),

            # Formulario + tarjeta lateral
            dbc.Row(
                [
                    # Formulario principal (col 8)
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div(
                                        [
                                            html.H5(
                                                "Características del paciente",
                                                style={
                                                    "fontWeight": "850",
                                                    "color": C_NAVY,
                                                    "fontSize": "1.15rem",
                                                    "marginBottom": "4px",
                                                },
                                            ),
                                            html.P(
                                                "Completa los 4 bloques con la información "
                                                "del perfil que deseas evaluar.",
                                                style={
                                                    "fontSize": "0.84rem",
                                                    "color": C_MUTED,
                                                    "marginBottom": "18px",
                                                },
                                            ),
                                        ]
                                    ),

                                    html.Div(_simulator_form_inputs()),

                                    # Botones de acción al final del formulario
                                    html.Div(
                                        [
                                            dbc.Button(
                                                [
                                                    html.I(className="fa-solid fa-play",
                                                           style={"marginRight": "8px"}),
                                                    html.Span("Predecir riesgo",
                                                              className="predict-btn-main-text"),
                                                ],
                                                id="sim-predict-button",
                                                n_clicks=0,
                                                className="predict-main-btn",
                                            ),
                                            dbc.Button(
                                                [
                                                    html.I(className="fa-solid fa-eraser",
                                                           style={"marginRight": "6px"}),
                                                    "Limpiar",
                                                ],
                                                id="sim-clear-button",
                                                n_clicks=0,
                                                className="predict-clear-btn",
                                            ),
                                        ],
                                        className="sim-action-buttons-bottom",
                                    ),
                                ],
                                className="p-4",
                            ),
                            className="border-0 shadow-sm",
                            style={
                                "borderRadius": "12px",
                                "backgroundColor": "#FFFFFF",
                            },
                        ),
                        md=12,
                        lg=8,
                    ),

                    # Sidebar: info modelo + resultado (col 4)
                    dbc.Col(
                        [
                            _model_info_card(),

                            html.Div(
                                id="sim-result-panel",
                                children=_default_result_panel(),
                                style={"marginTop": "18px"},
                            ),
                        ],
                        md=12,
                        lg=4,
                    ),
                ],
                className="g-3 mb-5 align-items-start",
            ),
        ]
    )


def _predict_subtab_content():
    """Contenido completo de la subpestaña 'Realizar predicción'."""
    return html.Div(
        [
            _simulator_block(),
        ],
        className="prediction-tab-content",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Layout principal (con subpestañas)
# ══════════════════════════════════════════════════════════════════════════════

def layout_model():
    """Layout completo de la pestaña /predict, con dos subpestañas."""
    return dbc.Container(
        [
            _hero(),

            # Subpestañas (Tabs de Bootstrap)
            dbc.Tabs(
                [
                    dbc.Tab(
                        _info_subtab_content(),
                        label="Información del modelo",
                        tab_id="tab-info",
                        labelClassName="prediction-tab-label",
                        activeLabelClassName="prediction-tab-label-active",
                    ),
                    dbc.Tab(
                        _predict_subtab_content(),
                        label="Realizar predicción",
                        tab_id="tab-predict",
                        labelClassName="prediction-tab-label",
                        activeLabelClassName="prediction-tab-label-active",
                    ),
                ],
                id="prediction-subtabs",
                active_tab="tab-info",
                className="prediction-tabs mb-4",
            ),

            html.Div(style={"height": "40px"}),  # padding inferior
        ],
        fluid=True,
        className="px-4 dashboard-shell model-page-shell",
        style={"minHeight": "100vh"},
    )
