from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# =========================
# PALETA
# =========================
C_NAVY  = "#0F2447"
C_BLUE  = "#1B3A6B"
C_RED   = "#C0392B"
C_BLUEI = "#4A7FC1"
C_BG    = "#F4F6F9"
C_TEXT  = "#2C3E50"
C_MUTED = "#7F8C8D"
C_BORDER= "#DDE2E8"
C_CARD  = "#FAFBFC"
C_SOFT  = "#EAF1FB"


# =========================
# CONFIGURACIÓN DE VARIABLES
# =========================
UNIVARIATE_CONFIG = {
    "Diabetes_binary":      {"label": "Diabetes (variable objetivo)",  "type": "categorica"},
    "HighBP":               {"label": "Hipertensión",                   "type": "categorica"},
    "HighChol":             {"label": "Colesterol alto",                "type": "categorica"},
    "CholCheck":            {"label": "Chequeo de colesterol",          "type": "categorica"},
    "BMI":                  {"label": "Índice de masa corporal (BMI)",  "type": "numerica"},
    "Smoker":               {"label": "Tabaquismo",                     "type": "categorica"},
    "Stroke":               {"label": "ACV",                            "type": "categorica"},
    "HeartDiseaseorAttack": {"label": "Enfermedad cardíaca",            "type": "categorica"},
    "PhysActivity":         {"label": "Actividad física",               "type": "categorica"},
    "Fruits":               {"label": "Consumo de frutas",              "type": "categorica"},
    "Veggies":              {"label": "Consumo de verduras",            "type": "categorica"},
    "HvyAlcoholConsump":    {"label": "Consumo alto de alcohol",        "type": "categorica"},
    "AnyHealthcare":        {"label": "Cobertura médica",               "type": "categorica"},
    "NoDocbcCost":          {"label": "Barrera económica de acceso",    "type": "categorica"},
    "GenHlth":              {"label": "Salud general percibida",        "type": "categorica"},
    "MentHlth":             {"label": "Días de mala salud mental",      "type": "numerica"},
    "PhysHlth":             {"label": "Días de mala salud física",      "type": "numerica"},
    "DiffWalk":             {"label": "Dificultad al caminar",          "type": "categorica"},
    "Sex":                  {"label": "Sexo",                           "type": "categorica"},
    "Age":                  {"label": "Edad",                           "type": "categorica"},
    "Education":            {"label": "Nivel educativo",                "type": "categorica"},
    "Income":               {"label": "Nivel de ingreso",               "type": "categorica"},
}

# Opciones para análisis bivariado
BINARY_VARS_BI  = [
    "HighBP","HighChol","CholCheck","Smoker","Stroke",
    "HeartDiseaseorAttack","PhysActivity","Fruits","Veggies",
    "HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","DiffWalk",
]
ORDINAL_VARS_BI = ["GenHlth","Age","Education","Income"]
NUMERIC_VARS_BI = ["BMI","MentHlth","PhysHlth"]

VAR_LABELS_BI = {
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
    [{"label": f"[Binaria] {VAR_LABELS_BI[v]}", "value": v} for v in BINARY_VARS_BI] +
    [{"label": f"[Ordinal] {VAR_LABELS_BI[v]}", "value": v} for v in ORDINAL_VARS_BI] +
    [{"label": f"[Numérica] {VAR_LABELS_BI[v]}", "value": v} for v in NUMERIC_VARS_BI]
)

# =========================
# DICCIONARIO DE DATOS
# =========================
DATA_DICTIONARY = [
    {"Variable": "Diabetes_binary",      "Rol": "Target",   "Tipo": "Binaria",               "Codificación": "Sin diabetes / Prediabetes–Diabetes",         "Descripción": "Indicador de presencia de diabetes o prediabetes"},
    {"Variable": "HighBP",               "Rol": "Feature",  "Tipo": "Binaria",               "Codificación": "Sin hipertensión / Con hipertensión",          "Descripción": "Diagnóstico de hipertensión arterial"},
    {"Variable": "HighChol",             "Rol": "Feature",  "Tipo": "Binaria",               "Codificación": "Sin colesterol alto / Con colesterol alto",    "Descripción": "Diagnóstico de colesterol elevado"},
    {"Variable": "CholCheck",            "Rol": "Feature",  "Tipo": "Binaria",               "Codificación": "Sin chequeo en 5 años / Con chequeo en 5 años","Descripción": "Realizó chequeo de colesterol en los últimos 5 años"},
    {"Variable": "Smoker",               "Rol": "Feature",  "Tipo": "Binaria",               "Codificación": "No fumador / Fumador",                         "Descripción": "Ha fumado al menos 100 cigarrillos en su vida"},
    {"Variable": "Stroke",               "Rol": "Feature",  "Tipo": "Binaria",               "Codificación": "Sin ACV / Con ACV",                            "Descripción": "Antecedente de accidente cerebrovascular"},
    {"Variable": "HeartDiseaseorAttack", "Rol": "Feature",  "Tipo": "Binaria",               "Codificación": "Sin enfermedad cardíaca / Con enfermedad cardíaca","Descripción": "Diagnóstico de enfermedad coronaria o infarto"},
    {"Variable": "PhysActivity",         "Rol": "Feature",  "Tipo": "Binaria",               "Codificación": "Sin actividad física / Con actividad física",  "Descripción": "Actividad física en los últimos 30 días (excluye trabajo)"},
    {"Variable": "Fruits",               "Rol": "Feature",  "Tipo": "Binaria",               "Codificación": "No consume frutas diario / Consume frutas diario","Descripción": "Consumo diario de frutas"},
    {"Variable": "Veggies",              "Rol": "Feature",  "Tipo": "Binaria",               "Codificación": "No consume verduras diario / Consume verduras diario","Descripción": "Consumo diario de verduras"},
    {"Variable": "HvyAlcoholConsump",    "Rol": "Feature",  "Tipo": "Binaria",               "Codificación": "No consumo alto alcohol / Consumo alto alcohol","Descripción": "Consumo elevado de alcohol según criterio BRFSS"},
    {"Variable": "AnyHealthcare",        "Rol": "Feature",  "Tipo": "Binaria",               "Codificación": "Sin cobertura médica / Con cobertura médica",  "Descripción": "Cuenta con seguro o cobertura médica"},
    {"Variable": "NoDocbcCost",          "Rol": "Feature",  "Tipo": "Binaria",               "Codificación": "Sin barrera económica / Con barrera económica","Descripción": "No pudo consultar médico por razones económicas"},
    {"Variable": "DiffWalk",             "Rol": "Feature",  "Tipo": "Binaria",               "Codificación": "Sin dificultad al caminar / Con dificultad",   "Descripción": "Dificultad para caminar o subir escaleras"},
    {"Variable": "Sex",                  "Rol": "Feature",  "Tipo": "Binaria",               "Codificación": "Mujer / Hombre",                               "Descripción": "Sexo biológico del encuestado"},
    {"Variable": "BMI",                  "Rol": "Feature",  "Tipo": "Numérica continua",     "Codificación": "Valor numérico",                               "Descripción": "Índice de Masa Corporal"},
    {"Variable": "MentHlth",             "Rol": "Feature",  "Tipo": "Numérica discreta",     "Codificación": "0–30 días",                                    "Descripción": "Número de días con mala salud mental en los últimos 30 días"},
    {"Variable": "PhysHlth",             "Rol": "Feature",  "Tipo": "Numérica discreta",     "Codificación": "0–30 días",                                    "Descripción": "Número de días con mala salud física en los últimos 30 días"},
    {"Variable": "GenHlth",              "Rol": "Feature",  "Tipo": "Ordinal",               "Codificación": "Excelente / Muy buena / Buena / Regular / Mala","Descripción": "Autopercepción del estado general de salud"},
    {"Variable": "Age",                  "Rol": "Feature",  "Tipo": "Ordinal",               "Codificación": "18–24 … 80+",                                 "Descripción": "Categoría de edad (13 grupos ordenados)"},
    {"Variable": "Education",            "Rol": "Feature",  "Tipo": "Ordinal",               "Codificación": "Nunca asistió … Universidad completa",         "Descripción": "Nivel educativo del encuestado"},
    {"Variable": "Income",               "Rol": "Feature",  "Tipo": "Ordinal",               "Codificación": "< $10,000 … $75,000+",                        "Descripción": "Categoría de ingresos anuales del hogar"},
]


# =========================
# LAYOUT PRINCIPAL
# =========================
def layout_about():
    return dbc.Container(
        [
            # ── ENCABEZADO ──────────────────────────────────────────────────
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.Span(
                                "Resumen del proyecto",
                                style={
                                    "display": "inline-block",
                                    "backgroundColor": C_SOFT,
                                    "color": C_BLUE,
                                    "padding": "6px 12px",
                                    "borderRadius": "999px",
                                    "fontSize": "0.78rem",
                                    "fontWeight": "700",
                                    "marginBottom": "14px",
                                },
                            ),
                            html.H2(
                                "Análisis exploratorio de indicadores de salud y factores asociados a la diabetes",
                                style={"fontWeight": "800", "color": C_NAVY,
                                       "marginBottom": "10px", "fontSize": "2rem"},
                            ),
                            html.P(
                                "Esta sección resume el contexto, los objetivos, la metodología, "
                                "los hallazgos más relevantes y el valor analítico del proyecto "
                                "a partir del dataset BRFSS 2015.",
                                style={"fontSize": "0.98rem", "color": C_MUTED,
                                       "maxWidth": "980px", "marginBottom": "20px"},
                            ),
                            dbc.Row(
                                dbc.Col(
                                    dbc.Button(
                                        "Ver notebook completo",
                                        href="https://paolacorr67-ctrl.github.io/machinedef/",
                                        target="_blank",
                                        style={"backgroundColor": C_BLUE,
                                               "border": "none", "fontWeight": "600"},
                                        className="me-2",
                                    ),
                                    xs=12, sm="auto",
                                ),
                                className="g-2",
                            ),
                        ]),
                        className="border-0 shadow-sm",
                        style={"borderRadius": "18px", "backgroundColor": "#FFFFFF"},
                    ),
                    width=12,
                ),
                className="pt-4 pb-3",
            ),

            # ── MÉTRICAS RÁPIDAS ────────────────────────────────────────────
            dbc.Row(
                [
                    dbc.Col(_metric_card("253.680", "Registros analizados", C_BLUE),  md=3, sm=6, xs=12),
                    dbc.Col(_metric_card("22",      "Variables incluidas",  C_BLUEI), md=3, sm=6, xs=12),
                    dbc.Col(_metric_card("1",       "Variable objetivo",    C_RED),   md=3, sm=6, xs=12),
                    dbc.Col(_metric_card("BRFSS 2015", "Fuente de datos",   "#7F8C8D"), md=3, sm=6, xs=12),
                ],
                className="g-3 mb-4",
            ),

            # ── CUERPO PRINCIPAL (tabs + sidebar) ───────────────────────────
            dbc.Row(
                [
                    # Columna principal — 8 cols
                    dbc.Col(
                        [
                            # Tabs de información general
                            dbc.Card(
                                dbc.CardBody([
                                    html.P("Resumen", style=_section_kicker()),
                                    html.H4(
                                        "Visión general del análisis",
                                        style={"fontWeight": "700", "color": C_NAVY, "marginBottom": "18px"},
                                    ),
                                    dbc.Tabs(
                                        [
                                            dbc.Tab(label="Contexto", tab_id="tab-contexto", label_style={"fontWeight": "600"}),
                                            dbc.Tab(label="Objetivos", tab_id="tab-objetivos", label_style={"fontWeight": "600"}),
                                            dbc.Tab(label="Propósito", tab_id="tab-proposito", label_style={"fontWeight": "600"}),
                                            dbc.Tab(label="Metodología", tab_id="tab-metodologia", label_style={"fontWeight": "600"}),
                                            dbc.Tab(label="Marco teórico", tab_id="tab-marco", label_style={"fontWeight": "600"}),
                                        ],
                                        id="about-main-tabs",
                                        active_tab="tab-contexto",
                                        className="mb-3",
                                    ),
                                    html.Div(id="about-main-tabs-content"),
                                ], className="p-4"),
                                className="border-0 shadow-sm mb-4",
                                style={"borderRadius": "16px", "backgroundColor": C_CARD},
                            ),

                            # Diccionario de datos
                            dbc.Card(
                                dbc.CardBody([
                                    html.P("Estructura de variables", style=_section_kicker()),
                                    html.H4(
                                        "Diccionario de datos",
                                        style={"fontWeight": "700", "color": C_NAVY, "marginBottom": "8px"},
                                    ),
                                    html.P(
                                        "La siguiente tabla resume el rol, tipo, codificación y descripción "
                                        "de las variables incluidas en el estudio.",
                                        style={"fontSize": "0.90rem", "color": C_MUTED,
                                               "marginBottom": "18px", "lineHeight": "1.8"},
                                    ),
                                    _data_dictionary_table(),
                                ], className="p-4"),
                                className="border-0 shadow-sm mb-4",
                                style={"borderRadius": "16px", "backgroundColor": C_CARD},
                            ),

                            # ── MÓDULO UNIVARIADO / BIVARIADO CON TABS ───────
                            dbc.Card(
                                dbc.CardBody([
                                    html.P("Módulo interactivo", className="section-kicker"),
                                    html.H4(
                                        "Exploración de variables",
                                        className="section-title",
                                        style={"fontSize": "1.35rem", "marginBottom": "8px"},
                                    ),
                                    html.P(
                                        "Alterna entre la lectura univariada y la comparación bivariada contra "
                                        "Diabetes_binary sin salir del mismo módulo de análisis.",
                                        className="section-subtitle",
                                        style={"fontSize": "0.92rem", "marginBottom": "20px"},
                                    ),
                                    dbc.Tabs(
                                        [
                                            dbc.Tab(
                                                label="Análisis univariado",
                                                tab_id="about-tab-univariate",
                                                children=html.Div(
                                                    [
                                                        dbc.Row(
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Variable a explorar",
                                                                        style={
                                                                            "fontSize": "0.82rem",
                                                                            "fontWeight": "700",
                                                                            "color": C_TEXT,
                                                                            "marginBottom": "8px",
                                                                        },
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="about-univariate-variable",
                                                                        options=[
                                                                            {"label": cfg["label"], "value": col}
                                                                            for col, cfg in UNIVARIATE_CONFIG.items()
                                                                        ],
                                                                        value="BMI",
                                                                        clearable=False,
                                                                        style={"fontSize": "0.9rem"},
                                                                    ),
                                                                ],
                                                                md=6,
                                                            ),
                                                            className="mb-3",
                                                        ),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    html.Div(
                                                                        dcc.Graph(
                                                                            id="about-univariate-graph",
                                                                            config={"displayModeBar": False},
                                                                            style={"height": "400px"},
                                                                        ),
                                                                        className="graph-card p-3 h-100",
                                                                    ),
                                                                    md=8,
                                                                ),
                                                                dbc.Col(
                                                                    html.Div(
                                                                        html.Div(id="about-univariate-summary"),
                                                                        className="premium-card h-100 p-4",
                                                                    ),
                                                                    md=4,
                                                                ),
                                                            ],
                                                            className="g-3",
                                                        ),
                                                    ],
                                                    className="analysis-pane",
                                                ),
                                            ),
                                            dbc.Tab(
                                                label="Análisis bivariado",
                                                tab_id="about-tab-bivariate",
                                                children=html.Div(
                                                    [
                                                        dbc.Row(
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Variable a comparar con Diabetes_binary",
                                                                        style={
                                                                            "fontSize": "0.82rem",
                                                                            "fontWeight": "700",
                                                                            "color": C_TEXT,
                                                                            "marginBottom": "8px",
                                                                        },
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="about-bivariate-variable",
                                                                        options=BIVARIATE_OPTIONS,
                                                                        value="HighBP",
                                                                        clearable=False,
                                                                        style={"fontSize": "0.9rem"},
                                                                    ),
                                                                ],
                                                                md=7,
                                                            ),
                                                            className="mb-3",
                                                        ),
                                                        html.Div(id="about-bivariate-type-badge", className="mb-3"),
                                                        html.Div(
                                                            dcc.Graph(
                                                                id="about-bivariate-main-graph",
                                                                config={"displayModeBar": False},
                                                                style={"height": "420px"},
                                                            ),
                                                            className="graph-card p-3 mb-3",
                                                        ),
                                                        html.Div(
                                                            id="about-bivariate-density-wrapper",
                                                            children=[
                                                                html.Div(
                                                                    dcc.Graph(
                                                                        id="about-bivariate-density-graph",
                                                                        config={"displayModeBar": False},
                                                                        style={"height": "360px"},
                                                                    ),
                                                                    className="graph-card p-3",
                                                                )
                                                            ],
                                                            style={"display": "none"},
                                                        ),
                                                        html.Div(id="about-bivariate-insight", className="mt-3"),
                                                    ],
                                                    className="analysis-pane",
                                                ),
                                            ),
                                        ],
                                        active_tab="about-tab-univariate",
                                        className="premium-tabs",
                                    ),
                                ], className="p-4"),
                                className="analysis-module mb-4",
                            ),
                        ],
                        md=8,
                    ),

                    # Columna sidebar — 4 cols
                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody([
                                    html.P("Resumen del dataset", style=_section_kicker()),
                                    html.H5("Panorama general",
                                            style={"fontWeight": "700", "color": C_NAVY, "marginBottom": "14px"}),
                                    _summary_line("Fuente",       "BRFSS 2015"),
                                    _summary_line("Observaciones","253.680"),
                                    _summary_line("Variables",    "22"),
                                    _summary_line("Target",       "Diabetes_binary"),
                                    _summary_line("Numéricas",    "BMI, MentHlth, PhysHlth"),
                                    _summary_line("Ordinales",    "GenHlth, Age, Education, Income"),
                                    html.Hr(style={"borderColor": C_BORDER}),
                                    html.P(
                                        "El dataset fue revisado en estructura, completitud y consistencia "
                                        "conceptual antes de construir la capa de visualización.",
                                        style={"fontSize": "0.86rem", "color": "#5D6D7E",
                                               "lineHeight": "1.7", "marginBottom": "0"},
                                    ),
                                ], className="p-4"),
                                className="border-0 shadow-sm mb-3",
                                style={"borderRadius": "16px", "backgroundColor": C_CARD},
                            ),

                            dbc.Card(
                                dbc.CardBody([
                                    html.P("Calidad de datos", style=_section_kicker()),
                                    html.H5("Verificaciones realizadas",
                                            style={"fontWeight": "700", "color": C_NAVY, "marginBottom": "14px"}),
                                    _insight_item("No se identificaron valores nulos en las variables analizadas."),
                                    _insight_item("Se revisaron duplicados y consistencia estructural del conjunto."),
                                    _insight_item("Las variables fueron reclasificadas en binarias, ordinales y "
                                                  "numéricas para una interpretación más clara."),
                                ], className="p-4"),
                                className="border-0 shadow-sm mb-3",
                                style={"borderRadius": "16px", "backgroundColor": C_CARD},
                            ),

                            dbc.Card(
                                dbc.CardBody([
                                    html.P("Guía de análisis bivariado", style=_section_kicker()),
                                    html.H5("Tipos de visualización",
                                            style={"fontWeight": "700", "color": C_NAVY, "marginBottom": "14px"}),
                                    _bivariate_guide_item(
                                        "Binaria vs. Diabetes",
                                        "Barras agrupadas — muestra el conteo de personas por categoría "
                                        "del factor dentro de cada grupo de diabetes.",
                                        C_BLUE,
                                    ),
                                    _bivariate_guide_item(
                                        "Ordinal vs. Diabetes",
                                        "Barras apiladas al 100% — facilita comparar la composición "
                                        "porcentual a lo largo del orden natural de la variable.",
                                        C_BLUEI,
                                    ),
                                    _bivariate_guide_item(
                                        "Numérica vs. Diabetes",
                                        "Boxplot + curva de densidad — el boxplot compara mediana y "
                                        "dispersión; la densidad muestra la forma de la distribución.",
                                        C_RED,
                                    ),
                                ], className="p-4"),
                                className="border-0 shadow-sm mb-3",
                                style={"borderRadius": "16px", "backgroundColor": C_CARD},
                            ),

                            dbc.Card(
                                dbc.CardBody([
                                    html.P("Recursos", style=_section_kicker()),
                                    html.H5("Material complementario",
                                            style={"fontWeight": "700", "color": C_NAVY, "marginBottom": "14px"}),
                                    html.A(
                                        "Abrir notebook completo",
                                        href="https://paolacorr67-ctrl.github.io/machinedef/",
                                        target="_blank",
                                        style=_resource_link_style(),
                                    ),
                                ], className="p-4"),
                                className="border-0 shadow-sm",
                                style={"borderRadius": "16px", "backgroundColor": C_CARD},
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="g-4 pb-5",
            ),
        ],
        fluid=True,
        className="analysis-shell px-3 px-md-4",
        style={"minHeight": "100vh"},
    )


# =========================
# HELPERS VISUALES
# =========================

def _metric_card(value, label, color):
    return dbc.Card(
        dbc.CardBody([
            html.Div(style={"width": "36px", "height": "4px", "backgroundColor": color,
                            "borderRadius": "3px", "marginBottom": "14px"}),
            html.H4(value, style={"fontWeight": "800", "color": C_NAVY, "marginBottom": "4px"}),
            html.P(label,  style={"fontSize": "0.86rem", "color": C_MUTED, "marginBottom": "0"}),
        ], className="p-4"),
        className="premium-card",
        style={"--kpi-accent": color},
    )


def _section_kicker():
    return {
        "fontSize": "0.72rem", "fontWeight": "700", "textTransform": "uppercase",
        "letterSpacing": "0.08em", "color": C_MUTED, "marginBottom": "6px",
    }


def _info_panel(title, body):
    return html.Div([
        html.H6(title, style={"fontWeight": "700", "color": C_TEXT, "marginBottom": "10px"}),
        html.P(body, style={"fontSize": "0.90rem", "color": "#5D6D7E",
                            "lineHeight": "1.9", "marginBottom": "0"}),
    ])


def _bullet(text):
    return html.Div([
        html.Div(style={"width": "9px", "height": "9px", "backgroundColor": C_BLUE,
                        "borderRadius": "50%", "marginTop": "8px", "marginRight": "10px",
                        "flexShrink": "0"}),
        html.P(text, style={"fontSize": "0.89rem", "color": "#5D6D7E",
                            "lineHeight": "1.8", "marginBottom": "8px"}),
    ], style={"display": "flex", "alignItems": "flex-start"})


def _insight_item(text):
    return html.Div([
        html.Div(style={"width": "10px", "height": "10px", "backgroundColor": C_BLUEI,
                        "borderRadius": "50%", "marginTop": "7px", "marginRight": "10px",
                        "flexShrink": "0"}),
        html.P(text, style={"fontSize": "0.87rem", "color": "#5D6D7E",
                            "lineHeight": "1.7", "marginBottom": "10px"}),
    ], style={"display": "flex", "alignItems": "flex-start"})


def _bivariate_guide_item(title, desc, color):
    return html.Div([
        html.Div([
            html.Div(style={"width": "12px", "height": "12px", "backgroundColor": color,
                            "borderRadius": "3px", "marginTop": "3px", "marginRight": "10px",
                            "flexShrink": "0"}),
            html.Div([
                html.Span(title, style={"fontSize": "0.83rem", "fontWeight": "700",
                                        "color": C_TEXT, "display": "block", "marginBottom": "3px"}),
                html.Span(desc,  style={"fontSize": "0.80rem", "color": "#5D6D7E", "lineHeight": "1.6"}),
            ]),
        ], style={"display": "flex", "alignItems": "flex-start"}),
    ], style={"marginBottom": "16px"})


def _summary_row(label, value):
    return html.Div([
        html.Span(label, style={"fontSize": "0.82rem", "fontWeight": "700", "color": C_TEXT}),
        html.Span(value, style={"fontSize": "0.82rem", "color": C_MUTED, "float": "right"}),
    ], style={"marginBottom": "10px"})


def _summary_line(label, value):
    return html.Div([
        html.Span(label, style={"fontSize": "0.82rem", "fontWeight": "700", "color": C_TEXT,
                                "display": "inline-block", "minWidth": "95px"}),
        html.Span(value, style={"fontSize": "0.82rem", "color": C_MUTED}),
    ], style={"marginBottom": "8px"})


def _resource_link_style():
    return {"display": "block", "fontWeight": "600", "color": C_BLUE,
            "textDecoration": "none", "marginBottom": "10px"}


def _mini_chip(text):
    return html.Span(text, style={
        "display": "inline-block", "backgroundColor": C_SOFT, "color": C_BLUE,
        "padding": "4px 10px", "borderRadius": "999px", "fontSize": "0.76rem",
        "fontWeight": "700", "marginRight": "8px", "marginBottom": "8px",
    })


def _process_box(title, items, bg_color, accent_color):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, style={
                "display": "inline-block", "padding": "6px 12px", "borderRadius": "999px",
                "backgroundColor": accent_color, "color": "white", "fontSize": "0.78rem",
                "fontWeight": "700", "letterSpacing": "0.02em", "marginBottom": "14px",
            }),
            html.H6(title, style={"fontWeight": "800", "color": C_NAVY, "marginBottom": "12px"}),
            html.Ul([
                html.Li(item, style={"marginBottom": "8px", "color": "#5D6D7E",
                                     "lineHeight": "1.75", "fontSize": "0.87rem"})
                for item in items
            ], style={"paddingLeft": "18px", "marginBottom": "0"}),
        ]),
        className="border-0 shadow-sm h-100",
        style={"borderRadius": "18px", "backgroundColor": bg_color,
               "border": f"1px solid {C_BORDER}"},
    )


def _objective_box(number, title, description, accent_color=C_BLUE):
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Div(number, style={
                    "width": "42px",
                    "height": "42px",
                    "borderRadius": "12px",
                    "backgroundColor": accent_color,
                    "color": "#FFFFFF",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "fontWeight": "800",
                    "fontSize": "0.90rem",
                    "marginBottom": "14px",
                    "boxShadow": f"0 8px 20px {accent_color}22",
                }),
                html.H6(title, style={
                    "fontWeight": "800",
                    "color": C_NAVY,
                    "marginBottom": "10px",
                    "lineHeight": "1.45"
                }),
                html.P(description, style={
                    "fontSize": "0.86rem",
                    "color": C_MUTED,
                    "lineHeight": "1.8",
                    "marginBottom": "0"
                }),
            ])
        ]),
        className="border-0 shadow-sm h-100",
        style={
            "borderRadius": "18px",
            "backgroundColor": C_CARD,
            "border": f"1px solid {C_BORDER}",
            "borderLeft": f"6px solid {accent_color}",
        },
    )

def _text_block():
    return {
        "fontSize": "0.9rem",
        "color": C_MUTED,
        "lineHeight": "1.85",
        "marginBottom": "10px",
    }


def _context_box(number, title, text, color):
    return dbc.Card(
        dbc.CardBody([
            html.Div(number, style={
                "width": "38px",
                "height": "38px",
                "borderRadius": "10px",
                "backgroundColor": color,
                "color": "white",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "fontWeight": "800",
                "marginBottom": "12px",
                "boxShadow": f"0 6px 14px {color}33",
            }),
            html.H6(title, style={
                "fontWeight": "800",
                "color": C_NAVY,
                "marginBottom": "8px",
            }),
            html.P(text, style={
                "fontSize": "0.85rem",
                "color": C_MUTED,
                "lineHeight": "1.7",
                "marginBottom": "0",
            }),
        ]),
        className="border-0 shadow-sm h-100",
        style={
            "borderRadius": "18px",
            "background": "white",
            "borderTop": f"4px solid {color}",
        },
    )


def _purpose_box(title, text, color):
    return dbc.Card(
        dbc.CardBody([
            html.Div(style={
                "width": "30px",
                "height": "4px",
                "backgroundColor": color,
                "borderRadius": "3px",
                "marginBottom": "10px",
            }),
            html.H6(title, style={
                "fontWeight": "800",
                "color": C_NAVY,
                "marginBottom": "6px",
            }),
            html.P(text, style={
                "fontSize": "0.85rem",
                "color": C_MUTED,
                "lineHeight": "1.7",
                "marginBottom": "0",
            }),
        ]),
        className="border-0 shadow-sm h-100",
        style={"borderRadius": "16px", "backgroundColor": "#FFFFFF"},
    )


def _theory_text_style():
    return {
        "fontSize": "0.89rem",
        "color": C_MUTED,
        "lineHeight": "1.8",
        "marginBottom": "0",
    }


def _formula_box(markdown_text):
    return html.Div(
        dcc.Markdown(
            markdown_text,
            mathjax=True,
            style={
                "fontSize": "0.86rem",
                "color": "#425466",
                "lineHeight": "1.8",
                "marginBottom": "0",
            },
        ),
        style={
            "backgroundColor": "#FFFFFF",
            "border": f"1px solid {C_BORDER}",
            "borderRadius": "14px",
            "padding": "14px 16px",
            "marginTop": "12px",
        },
    )


def _theory_box(section, title, intro, formula_markdown, bg_color, accent_color):
    return dbc.Card(
        dbc.CardBody([
            html.Div(section, style={
                "display": "inline-block",
                "padding": "6px 12px",
                "borderRadius": "999px",
                "backgroundColor": accent_color,
                "color": "white",
                "fontSize": "0.76rem",
                "fontWeight": "800",
                "letterSpacing": "0.04em",
                "marginBottom": "12px",
            }),
            html.H6(title, style={
                "fontWeight": "900",
                "color": C_NAVY,
                "marginBottom": "10px",
                "lineHeight": "1.4",
            }),
            html.P(intro, style={
                "fontSize": "0.86rem",
                "color": "#5D6D7E",
                "lineHeight": "1.75",
                "marginBottom": "0",
            }),
            _formula_box(formula_markdown),
        ]),
        className="border-0 shadow-sm h-100",
        style={
            "borderRadius": "18px",
            "backgroundColor": bg_color,
            "border": f"1px solid {C_BORDER}",
        },
    )


def _data_dictionary_table():
    header_style = {
        "backgroundColor": C_SOFT, "color": C_NAVY, "fontWeight": "700",
        "fontSize": "0.84rem", "borderBottom": f"1px solid {C_BORDER}", "whiteSpace": "nowrap",
    }
    cell_style = {
        "fontSize": "0.82rem", "color": "#5D6D7E", "verticalAlign": "top",
        "lineHeight": "1.6", "borderColor": C_BORDER, "minWidth": "140px",
    }
    thead = html.Thead(
        html.Tr([html.Th(col, style=header_style) for col in DATA_DICTIONARY[0].keys()])
    )
    rows = [
        html.Tr([
            html.Td(row["Variable"],     style={**cell_style, "fontWeight": "700", "color": C_TEXT}),
            html.Td(row["Rol"],          style=cell_style),
            html.Td(row["Tipo"],         style=cell_style),
            html.Td(row["Codificación"], style=cell_style),
            html.Td(row["Descripción"],  style={**cell_style, "minWidth": "260px"}),
        ])
        for row in DATA_DICTIONARY
    ]
    return html.Div(
        dbc.Table(
            [thead, html.Tbody(rows)],
            bordered=False, hover=True, responsive=True,
            className="mb-0 align-middle",
            style={"backgroundColor": "#FFFFFF", "borderCollapse": "separate", "borderSpacing": "0"},
        ),
        style={
            "border": f"1px solid {C_BORDER}", "borderRadius": "14px",
            "overflowX": "auto", "overflowY": "auto", "maxHeight": "560px",
            "backgroundColor": "#FFFFFF",
        },
    )


# =========================
# CALLBACKS
# =========================
def register_about_callbacks(app, df_viz: pd.DataFrame):

    # ── Tabs de información general ─────────────────────────────────────────
    @app.callback(
        Output("about-main-tabs-content", "children"),
        Input("about-main-tabs", "active_tab"),
    )
    def render_about_tab_content(active_tab):
        if active_tab == "tab-contexto":
            return html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                style={
                                    "width": "60px",
                                    "height": "4px",
                                    "background": "linear-gradient(90deg, #4A7FC1, #1B3A6B)",
                                    "borderRadius": "4px",
                                    "marginBottom": "14px",
                                }
                            ),
                            html.H4(
                                "Contexto del proyecto",
                                style={"fontWeight": "900", "color": C_NAVY, "marginBottom": "10px"},
                            ),
                            html.P(
                                "La diabetes constituye un problema prioritario de salud pública por su alta prevalencia "
                                "y su impacto sobre la calidad de vida.",
                                style={
                                    "fontSize": "0.95rem",
                                    "color": "#5D6D7E",
                                    "lineHeight": "1.8",
                                    "maxWidth": "850px",
                                },
                            ),
                        ],
                        style={"marginBottom": "28px"},
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    "PANORAMA GENERAL",
                                    style={
                                        "fontSize": "0.7rem",
                                        "fontWeight": "800",
                                        "letterSpacing": "0.08em",
                                        "color": "#7F8C8D",
                                        "marginBottom": "8px",
                                    },
                                ),
                                html.H4(
                                    "¿Por qué importa estudiar la diabetes?",
                                    style={"fontWeight": "900", "color": C_NAVY, "marginBottom": "12px"},
                                ),
                                html.P(
                                    "Es una condición metabólica en la que el organismo no produce insulina suficiente "
                                    "o no la utiliza correctamente, elevando los niveles de glucosa en sangre y generando "
                                    "riesgos progresivos para la salud.",
                                    style={
                                        "fontSize": "0.9rem",
                                        "color": C_MUTED,
                                        "lineHeight": "1.85",
                                        "marginBottom": "0",
                                    },
                                ),
                            ]
                        ),
                        className="border-0 shadow-sm mb-4",
                        style={
                            "borderRadius": "20px",
                            "background": "linear-gradient(135deg, #FFFFFF 0%, #EEF3FB 100%)",
                            "borderLeft": "6px solid #4A7FC1",
                        },
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                _context_box(
                                    "01",
                                    "Impacto",
                                    "La diabetes se asocia con complicaciones cardiovasculares, renales y funcionales, "
                                    "además de representar una carga importante para los sistemas de salud.",
                                    "#4A7FC1",
                                ),
                                md=6,
                            ),
                            dbc.Col(
                                _context_box(
                                    "02",
                                    "Enfoque del estudio",
                                    "Este análisis explora patrones entre variables conductuales, clínicas y sociodemográficas "
                                    "usando el dataset BRFSS 2015.",
                                    "#1B3A6B",
                                ),
                                md=6,
                            ),
                        ],
                        className="g-3 mb-4",
                    ),
                    html.Div(
                        [
                            html.Div(
                                "IDEA CLAVE",
                                style={
                                    "fontSize": "0.7rem",
                                    "fontWeight": "800",
                                    "letterSpacing": "0.08em",
                                    "color": "#7F8C8D",
                                    "marginBottom": "8px",
                                },
                            ),
                            html.H5(
                                "La diabetes es un fenómeno multivariable",
                                style={"fontWeight": "900", "color": C_NAVY, "marginBottom": "8px"},
                            ),
                            html.P(
                                "Su comprensión requiere analizar la interacción entre salud, hábitos de vida y contexto social.",
                                style={
                                    "fontSize": "0.9rem",
                                    "color": C_MUTED,
                                    "lineHeight": "1.8",
                                    "marginBottom": "0",
                                },
                            ),
                        ],
                        style={
                            "padding": "20px",
                            "borderRadius": "18px",
                            "background": "linear-gradient(135deg, #FFFFFF 0%, #F5F8FC 100%)",
                            "border": f"1px solid {C_BORDER}",
                        },
                    ),
                ],
                style={"padding": "10px 6px 20px 6px"},
            )

        elif active_tab == "tab-objetivos":
            return html.Div([
                html.Div([
                    html.H4("Objetivos del análisis",
                            style={"fontWeight": "800", "color": C_NAVY, "marginBottom": "12px"}),
                ], style={"padding": "8px 4px 22px 4px",
                          "borderBottom": f"1px solid {C_BORDER}", "marginBottom": "24px"}),

                dbc.Card(dbc.CardBody([
                    html.P("Objetivo general", style=_section_kicker()),
                    html.H5("Propósito central del análisis",
                            style={"fontWeight": "800", "color": C_NAVY, "marginBottom": "12px"}),
                    html.P(
                        "Explorar y analizar visualmente la relación entre variables demográficas, "
                        "hábitos de vida y condiciones de salud con la presencia de diabetes en la "
                        "población adulta de Estados Unidos, utilizando el conjunto de datos BRFSS 2015.",
                        style={"fontSize": "0.92rem", "color": C_MUTED, "lineHeight": "1.9", "marginBottom": "0"},
                    ),
                ]), className="border-0 shadow-sm mb-4",
                   style={"borderRadius": "18px", "background": "linear-gradient(180deg,#FFFFFF 0%,#FBFCFE 100%)"}),

                html.Div([
                    html.H5("Objetivos específicos",
                            style={"fontWeight": "800", "color": C_NAVY, "marginBottom": "10px"}),
                    html.P(
                        "Los objetivos específicos se estructuran como frentes complementarios de análisis, "
                        "abarcando descripción, comparación, detección de patrones y comprensión relacional.",
                        style={"fontSize": "0.92rem", "color": "#5D6D7E", "lineHeight": "1.8", "marginBottom": "20px"},
                    ),
                    dbc.Row([
                        dbc.Col(_objective_box(
                            "01",
                            "Distribución de la variable objetivo",
                            "Describir la distribución de la presencia o ausencia de diabetes dentro del conjunto de datos.",
                            C_BLUE
                        ), md=6, className="mb-3"),

                        dbc.Col(_objective_box(
                            "02",
                            "Comportamiento individual de variables",
                            "Analizar el comportamiento de las variables numéricas y categóricas del conjunto de datos.",
                            C_NAVY
                        ), md=6, className="mb-3"),

                        dbc.Col(_objective_box(
                            "03",
                            "Patrones y asociaciones visuales",
                            "Identificar patrones visuales y posibles asociaciones entre factores de riesgo y la diabetes.",
                            C_RED
                        ), md=6, className="mb-3"),

                        dbc.Col(_objective_box(
                            "04",
                            "Comparación entre grupos",
                            "Comparar grupos poblacionales según características como edad, educación, ingreso y sexo.",
                            C_BLUEI
                        ), md=6, className="mb-3"),

                        dbc.Col(_objective_box(
                            "05",
                            "Detección de tendencias relevantes",
                            "Detectar tendencias, contrastes y relaciones que aporten comprensión al fenómeno.",
                            C_TEXT
                        ), md=12, className="mb-3"),
                    ], className="g-3"),
                ], style={
                        "backgroundColor": C_BG,
                        "padding": "22px",
                        "borderRadius": "18px",
                        "border": f"1px solid {C_BORDER}",
                        "marginBottom": "18px"
                    }),
            ], style={"padding": "6px 2px 18px 2px"})

        elif active_tab == "tab-proposito":
            return html.Div(
                [
                    html.Div(
                        [
                            html.H4(
                                "Propósito del estudio",
                                style={"fontWeight": "900", "color": C_NAVY, "marginBottom": "10px"},
                            ),
                        ],
                        style={"marginBottom": "28px"},
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5(
                                    "Comprensión analítica del fenómeno",
                                    style={"fontWeight": "900", "color": C_NAVY, "marginBottom": "12px"},
                                ),
                                html.P(
                                    "El propósito de este análisis exploratorio es comprender cómo diferentes factores "
                                    "relacionados con el estilo de vida, la salud general y las características sociodemográficas "
                                    "se distribuyen en la población y cómo se asocian con la presencia de diabetes.",
                                    style=_text_block(),
                                ),
                                html.P(
                                    "A través de técnicas de visualización de datos, se busca identificar patrones, contrastes "
                                    "entre grupos y factores relevantes que contribuyan a una interpretación más clara del problema "
                                    "desde una perspectiva descriptiva.",
                                    style=_text_block(),
                                ),
                            ]
                        ),
                        className="border-0 shadow-sm mb-4",
                        style={
                            "borderRadius": "20px",
                            "background": "linear-gradient(135deg, #FFFFFF 0%, #EEF3FB 100%)",
                            "borderLeft": "6px solid #4A7FC1",
                        },
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                _purpose_box(
                                    "Enfoque",
                                    "Analizar relaciones entre variables conductuales, clínicas y sociodemográficas.",
                                    "#4A7FC1",
                                ),
                                md=4,
                            ),
                            dbc.Col(
                                _purpose_box(
                                    "Método",
                                    "Uso de visualización de datos para identificar patrones y contrastes.",
                                    "#1B3A6B",
                                ),
                                md=4,
                            ),
                            dbc.Col(
                                _purpose_box(
                                    "Resultado esperado",
                                    "Mejor comprensión del fenómeno y base para análisis predictivo posterior.",
                                    "#00A389",
                                ),
                                md=4,
                            ),
                        ],
                        className="g-3 mb-4",
                    ),
                ],
                style={"padding": "10px 6px 20px 6px"},
            )

        elif active_tab == "tab-metodologia":
            return html.Div([
                html.Div([
                    html.H4("Metodología",
                            style={"fontWeight": "800", "color": C_NAVY, "marginBottom": "12px"}),
                    html.P(
                        "El análisis se desarrolló como un proceso estructurado y progresivo, orientado a "
                        "garantizar la calidad de los datos y facilitar una interpretación clara de los resultados.",
                        style={"fontSize": "0.95rem", "color": "#5D6D7E", "lineHeight": "1.9",
                               "maxWidth": "980px", "marginBottom": "0"},
                    ),
                ], style={"padding": "8px 4px 22px 4px",
                          "borderBottom": f"1px solid {C_BORDER}", "marginBottom": "24px"}),

                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("Base de datos",
                                style={"fontWeight": "800", "color": C_NAVY, "marginBottom": "12px"}),
                        html.P(
                            "Se trabajó con una versión procesada del dataset BRFSS 2015, que integra "
                            "más de 250 mil registros y variables clínicas, conductuales y sociodemográficas.",
                            style={"fontSize": "0.88rem", "color": C_MUTED, "lineHeight": "1.85", "marginBottom": "14px"},
                        ),
                        html.Div([_mini_chip("Clínicas"), _mini_chip("Conductuales"), _mini_chip("Sociodemográficas")],
                                 style={"display": "flex", "flexWrap": "wrap", "gap": "8px"}),
                    ]), className="border-0 shadow-sm h-100",
                       style={"borderRadius": "18px", "background": "linear-gradient(180deg,#FFFFFF 0%,#FBFCFE 100%)"}),
                    md=6, className="mb-3"),

                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("Entorno de trabajo",
                                style={"fontWeight": "800", "color": C_NAVY, "marginBottom": "12px"}),
                        html.P(
                            "Análisis desarrollado en Python y Jupyter Notebook, integrando visualización, "
                            "estadística y documentación en un mismo flujo.",
                            style={"fontSize": "0.88rem", "color": C_MUTED, "lineHeight": "1.85", "marginBottom": "14px"},
                        ),
                        html.Div([
                            dbc.Badge("Python",         color="primary",   className="me-2 mb-2"),
                            dbc.Badge("Jupyter",        color="secondary", className="me-2 mb-2"),
                            dbc.Badge("Pandas",         color="light", text_color="dark", className="me-2 mb-2"),
                            dbc.Badge("Plotly / Dash",  color="light", text_color="dark", className="me-2 mb-2"),
                            dbc.Badge("NumPy",          color="light", text_color="dark", className="me-2 mb-2"),
                            dbc.Badge("SciPy",          color="light", text_color="dark", className="me-2 mb-2"),
                        ]),
                    ]), className="border-0 shadow-sm h-100",
                       style={"borderRadius": "18px", "background": "linear-gradient(180deg,#FFFFFF 0%,#FBFCFE 100%)"}),
                    md=6, className="mb-3"),
                ], className="g-3 mb-4"),

                html.Div([
                    html.H5("Proceso de análisis",
                            style={"fontWeight": "800", "color": C_NAVY, "marginBottom": "10px"}),
                    dbc.Row([
                        dbc.Col(_process_box("Exploración inicial",
                            ["Revisión de la estructura del dataset.",
                             "Clasificación de variables según su naturaleza.",
                             "Verificación de calidad y consistencia de datos."],
                            "#EEF4FF",C_NAVY), md=6, className="mb-3"),
                        dbc.Col(_process_box("Detección de anomalías",
                            ["Identificación de valores atípicos en variables numéricas.",
                             "Evaluación de su impacto sobre los resultados del análisis."],
                            "#EEF4FF",C_RED), md=6, className="mb-3"),
                        dbc.Col(_process_box("Análisis descriptivo (univariado)",
                            ["Estudio del comportamiento individual de cada variable.",
                             "Estadísticas descriptivas y visualizaciones de distribución.",
                             "Comprensión de patrones, asimetrías y tendencias generales."],
                            "#EEF4FF", C_BLUE), md=6, className="mb-3"),
                        dbc.Col(_process_box("Análisis relacional (bivariado)",
                            ["Evaluación de relaciones entre cada variable y la presencia de diabetes.",
                             "Visualizaciones adaptadas según el tipo de variable.",
                             "Interpretación de diferencias entre grupos con y sin diabetes."],
                            "#EEF4FF",C_BLUEI), md=6, className="mb-3"),
                    ], className="g-3"),
                ], style={"backgroundColor": "#F8FAFC", "padding": "22px", "borderRadius": "18px",
                          "border": f"1px solid {C_BORDER}"}),
            ], style={"padding": "6px 2px 18px 2px"})

        elif active_tab == "tab-marco":
            return html.Div(
                [
                    html.Div(
                        [
                            html.H4(
                                "Marco teórico",
                                style={"fontWeight": "900", "color": C_NAVY, "marginBottom": "10px"},
                            ),
                        ],
                        style={
                            "padding": "8px 4px 22px 4px",
                            "borderBottom": f"1px solid {C_BORDER}",
                            "marginBottom": "24px",
                        },
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.P("Fundamento conceptual", style=_section_kicker()),
                                html.H5(
                                    "Bases estadísticas del estudio",
                                    style={"fontWeight": "900", "color": C_NAVY, "marginBottom": "12px"},
                                ),
                                html.P(
                                    "El análisis exploratorio de datos requiere herramientas de estadística descriptiva, "
                                    "inferencia y contrastación de hipótesis para resumir variables, detectar patrones, "
                                    "evaluar asociaciones y respaldar interpretaciones con criterios formales.",
                                    style=_theory_text_style(),
                                ),
                            ]
                        ),
                        className="border-0 shadow-sm mb-4",
                        style={
                            "borderRadius": "20px",
                            "background": "linear-gradient(135deg, #FFFFFF 0%, #EEF3FB 100%)",
                            "borderLeft": "6px solid #4A7FC1",
                        },
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                _theory_box(
                                    "4.1",
                                    "Estadística descriptiva",
                                    "Resume las características principales de las variables mediante medidas de tendencia central y dispersión.",
                                    r"""
La media aritmética se define como:

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

La varianza está dada por:

$$
\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2
$$

Su raíz cuadrada corresponde a la desviación estándar.
""",
                                    "#EEF4FF",
                                    "#5B6CFF",
                                ),
                                md=6,
                                className="mb-3",
                            ),
                            dbc.Col(
                                _theory_box(
                                    "4.2",
                                    "Distribución y detección de atípicos",
                                    "Permite evaluar la forma de los datos, su asimetría y la presencia de observaciones extremas.",
                                    r"""
Una técnica frecuente es el rango intercuartílico:

$$
IQR = Q_3 - Q_1
$$

Un valor se considera atípico si:

$$
x < Q_1 - 1.5 \cdot IQR \quad \text{o} \quad x > Q_3 + 1.5 \cdot IQR
$$
""",
                                    "#F1F5F9",
                                    "#64748B",
                                ),
                                md=6,
                                className="mb-3",
                            ),
                            dbc.Col(
                                _theory_box(
                                    "4.3",
                                    "Correlación lineal",
                                    "Evalúa la intensidad y dirección de la relación entre variables numéricas.",
                                    r"""
El coeficiente de correlación de Pearson se define como:

$$
r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sigma_x \sigma_y}
$$

Toma valores en el intervalo [-1, 1].
""",
                                    "#F1F5F9",
                                    "#64748B",
                                ),
                                md=6,
                                className="mb-3",
                            ),
                            dbc.Col(
                                _theory_box(
                                    "4.4",
                                    "Prueba Chi-cuadrado de independencia",
                                    "Se utiliza para evaluar asociación entre variables categóricas.",
                                    r"""
La estadística Chi-cuadrado se define como:

$$
\chi^2 = \sum \frac{(O - E)^2}{E}
$$

O son las frecuencias observadas y E las esperadas bajo independencia.
""",
                                    "#EEF4FF",
                                    "#5B6CFF",
                                ),
                                md=6,
                                className="mb-3",
                            ),
                            dbc.Col(
                                _theory_box(
                                    "4.5",
                                    "Pruebas de comparación de medias",
                                    "Permiten contrastar diferencias entre grupos cuando se analiza una variable numérica frente a una variable categórica binaria.",
                                    r"""
Una de las estadísticas más utilizadas es el t-test de Student:

$$
t = \frac{\bar{x}_1 - \bar{x}_2}{SE}
$$

Cuando no se cumple el supuesto de normalidad, puede emplearse la prueba no paramétrica de Mann-Whitney U.
""",
                                    "#EEF4FF",
                                    "#5B6CFF",
                                ),
                                md=6,
                                className="mb-3",
                            ),
                            dbc.Col(
                                _theory_box(
                                    "4.6",
                                    "Inferencia estadística y valor p",
                                    "La inferencia permite generalizar resultados de una muestra a una población.",
                                    r"""
El valor p representa la probabilidad de observar resultados al menos tan extremos como los obtenidos bajo la hipótesis nula.

El criterio de decisión más habitual es:

$$
p < 0.05 \Rightarrow \text{rechazo de } H_0
$$
""",
                                    "#F1F5F9",
                                    "#64748B",
                                ),
                                md=6,
                                className="mb-3",
                            ),
                        ],
                        className="g-3",
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H6(
                                    "Síntesis conceptual",
                                    style={"fontWeight": "900", "color": C_NAVY, "marginBottom": "10px"},
                                ),
                                html.P(
                                    "Estas herramientas proporcionan la base para describir, contrastar e interpretar "
                                    "el dataset de manera rigurosa, facilitando una lectura estadística sólida del fenómeno estudiado.",
                                    style=_theory_text_style(),
                                ),
                            ]
                        ),
                        className="border-0 shadow-sm mt-2",
                        style={
                            "borderRadius": "18px",
                            "background": "linear-gradient(135deg, #FFFFFF 0%, #F5F8FC 100%)",
                        },
                    ),
                ],
                style={"padding": "10px 6px 20px 6px"},
            )

        return html.Div()

    # ── Análisis UNIVARIADO ─────────────────────────────────────────────────
    @app.callback(
        Output("about-univariate-graph",   "figure"),
        Output("about-univariate-summary", "children"),
        Input("about-univariate-variable", "value"),
    )
    def update_univariate_view(selected_var):
        cfg = UNIVARIATE_CONFIG.get(selected_var, {"label": selected_var, "type": "categorica"})

        if selected_var not in df_viz.columns:
            fig = go.Figure()
            fig.update_layout(template="plotly_white", paper_bgcolor="white",
                              plot_bgcolor="white", margin=dict(l=20,r=20,t=50,b=20))
            return fig, html.P("Variable no disponible.", style={"fontSize": "0.86rem", "color": C_MUTED})

        series = df_viz[selected_var].dropna()

        if cfg["type"] == "numerica":
            fig = px.histogram(
                df_viz, x=selected_var, nbins=30,
                title=f"Distribución de {cfg['label']}",
                color_discrete_sequence=[C_BLUE],
            )
            fig.update_layout(
                template="plotly_white",
                margin=dict(l=22,r=20,t=66,b=44),
                title_font=dict(size=15, color=C_NAVY),
                paper_bgcolor="white", plot_bgcolor="white",
                xaxis_title=cfg["label"], yaxis_title="Frecuencia",
                bargap=0.06,
                hoverlabel=dict(
                    bgcolor="#0B1628",
                    bordercolor="#0B1628",
                    font=dict(color="#FFFFFF", size=12),
                ),
            )
            fig.update_xaxes(gridcolor="#E5EAF2", linecolor="#CBD5E1", tickfont=dict(color="#475569"), automargin=True)
            fig.update_yaxes(gridcolor="#E5EAF2", linecolor="#CBD5E1", tickfont=dict(color="#475569"), automargin=True)
            fig.update_traces(
                marker_line_width=0,
                opacity=0.9,
                hovertemplate=f"{cfg['label']}: %{{x}}<br>Frecuencia: %{{y:,}}<extra></extra>",
            )

            summary = html.Div([
                html.H6("Resumen estadístico", style={"fontWeight":"700","color":C_NAVY}),
                _summary_row("Tipo",          "Numérica"),
                _summary_row("Media",         f"{series.mean():.2f}"),
                _summary_row("Mediana",       f"{series.median():.2f}"),
                _summary_row("Desv. estándar",f"{series.std():.2f}"),
                _summary_row("Mínimo",        f"{series.min():.2f}"),
                _summary_row("Máximo",        f"{series.max():.2f}"),
                html.Hr(style={"borderColor": C_BORDER}),
                html.P(
                    "Esta visualización permite observar concentración, dispersión y posibles "
                    "asimetrías en la distribución de la variable seleccionada.",
                    style={"fontSize":"0.84rem","color":C_MUTED,"lineHeight":"1.7","marginBottom":"0"},
                ),
            ])
            return fig, summary

        # Categórica / ordinal
        counts = (
            series.astype("string")
            .value_counts(dropna=False, sort=False)
            .reset_index()
        )
        counts.columns = [selected_var, "Frecuencia"]
        counts["Porcentaje"] = counts["Frecuencia"] / counts["Frecuencia"].sum()
        x_order = counts[selected_var].tolist()

        fig = px.bar(
            counts, x=selected_var, y="Frecuencia",
            title=f"Distribución de {cfg['label']}",
            category_orders={selected_var: x_order},
            custom_data=["Porcentaje"],
            color_discrete_sequence=[C_BLUE],
        )
        tickangle = -30 if len(x_order) > 5 else 0
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=22,r=20,t=66,b=86 if tickangle else 44),
            title_font=dict(size=15, color=C_NAVY),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis_title="", yaxis_title="Frecuencia",
            bargap=0.22,
            hoverlabel=dict(
                bgcolor="#0B1628",
                bordercolor="#0B1628",
                font=dict(color="#FFFFFF", size=12),
            ),
        )
        fig.update_xaxes(
            gridcolor="#E5EAF2",
            linecolor="#CBD5E1",
            tickfont=dict(color="#475569"),
            tickangle=tickangle,
            automargin=True,
        )
        fig.update_yaxes(gridcolor="#E5EAF2", linecolor="#CBD5E1", tickfont=dict(color="#475569"), automargin=True)
        fig.update_traces(
            text=None,
            texttemplate=None,
            marker_line_width=0,
            opacity=0.92,
            hovertemplate="%{x}<br>Frecuencia: %{y:,}<br>Porcentaje: %{customdata[0]:.1%}<extra></extra>",
        )

        moda = series.mode().iloc[0] if not series.mode().empty else "N/A"
        summary = html.Div([
            html.H6("Resumen estadístico", style={"fontWeight":"700","color":C_NAVY}),
            _summary_row("Tipo",         "Categórica"),
            _summary_row("Categorías",   str(series.nunique())),
            _summary_row("Moda",         str(moda)),
            _summary_row("Observaciones",f"{len(series):,}"),
            html.Hr(style={"borderColor": C_BORDER}),
            html.P(
                "Esta visualización muestra cómo se distribuyen las categorías y permite "
                "detectar concentraciones, desbalances o predominio de ciertas clases.",
                style={"fontSize":"0.84rem","color":C_MUTED,"lineHeight":"1.7","marginBottom":"0"},
            ),
        ])
        return fig, summary

    # ── Análisis BIVARIADO ──────────────────────────────────────────────────
    @app.callback(
        Output("about-bivariate-main-graph",       "figure"),
        Output("about-bivariate-density-graph",    "figure"),
        Output("about-bivariate-density-wrapper",  "style"),
        Output("about-bivariate-type-badge",       "children"),
        Output("about-bivariate-insight",          "children"),
        Input("about-bivariate-variable",          "value"),
    )
    def update_bivariate_view(selected_var):
        from src.utils.figures import (
            fig_bivariate_binary,
            fig_bivariate_ordinal,
            fig_bivariate_numeric,
            BINARY_VARS,
            ORDINAL_VARS_BI,
            NUMERIC_VARS_BI,
        )

        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_white", paper_bgcolor="white",
                                plot_bgcolor="white", margin=dict(l=20,r=20,t=40,b=20))

        if not selected_var or selected_var not in df_viz.columns:
            return empty_fig, empty_fig, {"display": "none"}, html.Div(), html.Div()

        label = VAR_LABELS_BI.get(selected_var, selected_var)

        # Badge del tipo de variable
        if selected_var in BINARY_VARS:
            var_type  = "Binaria"
            badge_col = C_BLUE
            graph_desc = "Barras agrupadas — conteo de personas por categoría en cada grupo de diabetes."
        elif selected_var in ORDINAL_VARS_BI:
            var_type  = "Ordinal"
            badge_col = C_BLUEI
            graph_desc = "Barras apiladas al 100% — composición porcentual a lo largo del orden natural."
        else:
            var_type  = "Numérica"
            badge_col = C_RED
            graph_desc = "Boxplot + curva de densidad — distribución y forma por grupo de diabetes."

        badge = html.Div([
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

        density_style = {"display": "none"}
        density_fig   = empty_fig

        if selected_var in BINARY_VARS:
            main_fig = fig_bivariate_binary(df_viz, selected_var)
            insight  = _build_binary_insight(df_viz, selected_var, label)

        elif selected_var in ORDINAL_VARS_BI:
            main_fig = fig_bivariate_ordinal(df_viz, selected_var)
            insight  = _build_ordinal_insight(df_viz, selected_var, label)

        else:  # Numérica
            main_fig, density_fig = fig_bivariate_numeric(df_viz, selected_var)
            density_style = {"display": "block"}
            insight = _build_numeric_insight(df_viz, selected_var, label)

        return main_fig, density_fig, density_style, badge, insight


def _build_binary_insight(df_viz, var, label):
    """Genera panel de lectura para variable binaria."""
    try:
        dm   = df_viz["Diabetes_binary"] == "Prediabetes / Diabetes"
        cats = df_viz[var].dropna().unique()
        lines = []
        for cat in cats:
            mask  = df_viz[var] == cat
            pct_dm = (df_viz[dm & mask].shape[0] / df_viz[mask].shape[0] * 100) if df_viz[mask].shape[0] > 0 else 0
            lines.append(f"**{cat}:** {pct_dm:.1f}% con diabetes")
        text = "\n\n".join(lines) if lines else "Sin datos disponibles."
    except Exception:
        text = "Sin datos disponibles."

    return _insight_card(f"Lectura del análisis — {label}", text)


def _build_ordinal_insight(df_viz, var, label):
    """Genera panel de lectura para variable ordinal."""
    try:
        cross = pd.crosstab(df_viz[var], df_viz["Diabetes_binary"])
        if "Prediabetes / Diabetes" in cross.columns and "Sin diabetes" in cross.columns:
            totals  = cross.sum(axis=1)
            pct_dm  = (cross["Prediabetes / Diabetes"] / totals * 100).round(1)
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

    return _insight_card(f"Lectura del análisis — {label}", text)


def _build_numeric_insight(df_viz, var, label):
    """Genera panel de lectura para variable numérica."""
    try:
        dm   = df_viz["Diabetes_binary"] == "Prediabetes / Diabetes"
        v_si = df_viz[dm][var].dropna()
        v_no = df_viz[~dm][var].dropna()
        diff = v_si.mean() - v_no.mean()
        text = (
            f"Media en personas **con diabetes:** {v_si.mean():.2f} — "
            f"**sin diabetes:** {v_no.mean():.2f}. "
            f"Diferencia de {abs(diff):.2f} unidades "
            f"({'mayor' if diff > 0 else 'menor'} en el grupo con diabetes)."
        )
    except Exception:
        text = "Sin datos disponibles."

    return _insight_card(f"Lectura del análisis — {label}", text)


def _insight_card(title, text):
    return dbc.Card(
        dbc.CardBody([
            html.P(title, style={
                "fontSize": "0.72rem", "fontWeight": "700", "textTransform": "uppercase",
                "letterSpacing": "0.08em", "color": C_MUTED, "marginBottom": "8px",
            }),
            dcc.Markdown(text, style={
                "fontSize": "0.90rem", "lineHeight": "1.85",
                "color": "#34495E", "marginBottom": "0",
            }),
        ], className="px-4 py-3"),
        className="insight-card",
        style={"backgroundColor": C_CARD},
    )
