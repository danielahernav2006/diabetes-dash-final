"""
Callbacks para la pestaña Modelo Predictivo.

Cada callback tiene IDs prefijados ("rec-", "exp-", "sim-", "ml-", "cm-") para
evitar colisiones con los del dashboard EDA.

CAMBIOS RESPECTO A LA VERSIÓN ANTERIOR
--------------------------------------
1. `_build_result_panel`  →  rediseñado como "risk assessment card" con
   banner superior coloreado, valor de probabilidad grande, barra de
   progreso visual y cuerpo estructurado. Usa las clases CSS .result-* y
   .result-tag.is-positive / .is-negative.
2. Nuevo callback `render_cm_mini_cards`  →  rellena las mini tarjetas
   TN / FP / FN / TP que acompañan a la matriz de confusión. Usa las
   clases CSS .cm-mini-card y variantes (.is-good, .is-warn, .is-bad).
3. Nuevo callback `clear_simulator_form`  →  reinicia los 21 inputs del
   formulario al pulsar el botón "Limpiar".
4. Nuevo callback `reset_result_panel_on_clear`  →  vuelve el panel de
   resultado al estado inicial cuando se pulsa "Limpiar".
"""

from dash import Input, Output, State, html, dcc, no_update, callback_context
from dash.dash_table import DataTable
import dash_bootstrap_components as dbc

from src.utils.model_utils import (
    MODEL_DESCRIPTIONS, MODEL_LABELS_ES,
    fmt_pct, get_best_model_name, get_confusion_matrix, get_metrics_df,
    predict_with_model,
)
from src.utils.model_figures import (
    fig_confusion_matrix, fig_feature_importance, fig_metrics_comparison,
    fig_pr_curves, fig_roc_curves,
)


# ── Paleta consistente ───────────────────────────────────────────────────────
C_NAVY  = "#0F2447"
C_BLUE  = "#1B3A6B"
C_RED   = "#C0392B"
C_BLUEI = "#4A7FC1"
C_TEXT  = "#2C3E50"
C_MUTED = "#7F8C8D"
C_BORDER = "#DDE2E8"
C_SOFT  = "#EAF1FB"
C_GREEN = "#1E8449"


# ── Valores por defecto del simulador (deben coincidir con model.py) ─────────
SIM_DEFAULTS = {
    "Sex": 0, "Age": 9, "Education": 5, "Income": 6,
    "HighBP": 0, "HighChol": 0, "CholCheck": 1,
    "Stroke": 0, "HeartDiseaseorAttack": 0, "BMI": 27,
    "Smoker": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "GenHlth": 3, "DiffWalk": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "MentHlth": 0, "PhysHlth": 0,
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# Comentario corto interpretativo según el modelo (acompaña los KPIs dinámicos)
def _interpretation_text(name: str, row: dict) -> str:
    f1 = row.get("f1", 0) or 0
    rec = row.get("recall", 0) or 0
    prec = row.get("precision", 0) or 0
    if name == "Random Forest":
        return (
            f"Equilibra recall ({rec*100:.1f}%) y precision ({prec*100:.1f}%) "
            f"y obtiene el F1 más alto ({f1*100:.1f}%). Es el modelo recomendado."
        )
    if name == "XGBoost":
        return (
            f"Recall muy alto ({rec*100:.1f}%): detecta gran parte de los "
            f"positivos, aunque con precision moderada ({prec*100:.1f}%)."
        )
    if name == "Logistic Regression":
        return (
            "Modelo lineal con `class_weight='balanced'`. Es interpretable y "
            f"alcanza un recall de {rec*100:.1f}%, útil como base de comparación."
        )
    if name == "KNN":
        return (
            f"Accuracy aparentemente alto, pero recall muy bajo ({rec*100:.1f}%): "
            "no detecta bien la clase positiva en este dataset desbalanceado."
        )
    return f"F1: {f1*100:.1f}% · Recall: {rec*100:.1f}% · Precision: {prec*100:.1f}%."


def _models_table(df):
    """Construye una DataTable con formato visual y la fila del mejor modelo destacada."""
    if df.empty:
        return html.Div("No hay métricas disponibles.",
                        style={"color": C_MUTED, "fontStyle": "italic"})

    df = df.copy()
    df["accuracy_fmt"]  = (df["accuracy"]  * 100).round(2).astype(str) + "%"
    df["precision_fmt"] = (df["precision"] * 100).round(2).astype(str) + "%"
    df["recall_fmt"]    = (df["recall"]    * 100).round(2).astype(str) + "%"
    df["f1_fmt"]        = (df["f1"]        * 100).round(2).astype(str) + "%"
    df["roc_fmt"]       = (df["roc_auc"]   * 100).round(2).astype(str) + "%"
    df["pr_fmt"]        = (df["pr_auc"]    * 100).round(2).astype(str) + "%"

    df["model_label"] = df["model"].map(MODEL_LABELS_ES).fillna(df["model"])

    best = get_best_model_name()

    columns = [
        {"name": "Modelo",     "id": "model_label"},
        {"name": "Accuracy",   "id": "accuracy_fmt"},
        {"name": "Precision",  "id": "precision_fmt"},
        {"name": "Recall",     "id": "recall_fmt"},
        {"name": "F1-score",   "id": "f1_fmt"},
        {"name": "ROC-AUC",    "id": "roc_fmt"},
        {"name": "PR-AUC",     "id": "pr_fmt"},
    ]

    return DataTable(
        columns=columns,
        data=df.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#F1F5F9",
            "color": C_NAVY,
            "fontWeight": "800",
            "fontSize": "11.5px",
            "textTransform": "uppercase",
            "letterSpacing": "0.04em",
            "border": f"1px solid {C_BORDER}",
            "padding": "10px 12px",
        },
        style_cell={
            "fontFamily": "Inter, Arial, sans-serif",
            "fontSize": "13px",
            "color": C_TEXT,
            "textAlign": "left",
            "padding": "10px 12px",
            "border": f"1px solid {C_BORDER}",
            "backgroundColor": "#FFFFFF",
        },
        style_cell_conditional=(
            [{"if": {"column_id": "model_label"},
              "fontWeight": "700", "color": C_NAVY}]
            + [{"if": {"column_id": c}, "textAlign": "center"}
               for c in ["accuracy_fmt", "precision_fmt", "recall_fmt",
                         "f1_fmt", "roc_fmt", "pr_fmt"]]
        ),
        style_data_conditional=[
            {
                "if": {"filter_query": f'{{model}} = "{best}"'},
                "backgroundColor": C_SOFT,
                "fontWeight": "700",
                "color": C_NAVY,
            },
        ],
        cell_selectable=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Panel de resultado de la simulación (rediseñado tipo "risk assessment card")
# ──────────────────────────────────────────────────────────────────────────────

def _initial_result_panel():
    """Estado inicial / al limpiar: replica el panel definido en model.py
    sin importar el módulo (para no crear dependencias circulares)."""
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
                            "formulario y presione “Predecir riesgo” para "
                            "obtener la clasificación estimada y la "
                            "probabilidad asociada.",
                            style={"fontSize": "0.88rem", "color": "#5D6D7E",
                                   "lineHeight": "1.7", "marginTop": "4px",
                                   "marginBottom": "10px"},
                        ),
                        html.P("Próximos pasos",
                               className="result-section-title"),
                        html.Ul(
                            [
                                html.Li("Completa los 4 grupos del formulario."),
                                html.Li("Elige el modelo a utilizar."),
                                html.Li("Ejecuta la predicción."),
                            ],
                            style={"fontSize": "0.84rem", "color": "#5D6D7E",
                                   "lineHeight": "1.7", "marginBottom": "0",
                                   "paddingLeft": "18px"},
                        ),
                        html.Div(
                            "Esta herramienta no constituye un diagnóstico "
                            "médico. Sus resultados se basan en patrones "
                            "observados en el dataset BRFSS 2015 y deben "
                            "interpretarse con criterio académico.",
                            className="result-disclaimer",
                        ),
                    ],
                    className="result-body",
                ),
            ],
            className="result-card",
        ),
    )


def _build_result_panel(model_name, label, prob):
    """Construye el panel de resultado de la simulación, rediseñado como
    una "risk assessment card" con banner coloreado, valor grande de
    probabilidad y barra visual."""

    # Caso de error (no se pudo calcular)
    if label is None:
        return html.Div(
            dbc.Card(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.P("Error en la predicción",
                                           className="result-banner-kicker"),
                                    html.H5("No fue posible calcular",
                                            className="result-banner-title"),
                                ]
                            ),
                        ],
                        className="result-card-banner",
                        style={
                            "background":
                                "linear-gradient(135deg, #6B5B27 0%, #C98A16 100%)",
                        },
                    ),
                    html.Div(
                        [
                            html.P(
                                "Verifica que el archivo del modelo esté disponible "
                                "en la carpeta `models/`. Puedes regenerarlo "
                                "ejecutando `python scripts/train_models.py`.",
                                style={"fontSize": "0.88rem", "color": "#5D6D7E",
                                       "lineHeight": "1.7", "marginBottom": "0"},
                            ),
                        ],
                        className="result-body",
                    ),
                ],
                className="result-card",
            ),
        )

    is_positive = (label == 1)

    # Color y etiquetas según el resultado
    if is_positive:
        banner_class = "result-card-banner is-positive"
        banner_kicker = "Resultado clínico estimado"
        banner_title = "Riesgo de prediabetes / diabetes"
        tag_class = "result-tag is-positive"
        tag_text = "Clase positiva"
        prob_color = C_RED
        prob_bg_grad = "linear-gradient(90deg, #B83A32 0%, #E07368 100%)"
        message = (
            "El modelo clasifica este perfil dentro del grupo con prediabetes/"
            "diabetes. Este resultado **no representa un diagnóstico médico**, "
            "sino una estimación basada en patrones del dataset BRFSS 2015."
        )
    else:
        banner_class = "result-card-banner is-negative"
        banner_kicker = "Resultado clínico estimado"
        banner_title = "Sin diabetes (clase negativa)"
        tag_class = "result-tag is-negative"
        tag_text = "Clase negativa"
        prob_color = C_GREEN
        prob_bg_grad = "linear-gradient(90deg, #1E8449 0%, #5BB57F 100%)"
        message = (
            "El modelo no clasifica este perfil dentro del grupo de prediabetes/"
            "diabetes. Sin embargo, esta estimación **no reemplaza una "
            "valoración médica** y depende exclusivamente de los datos provistos."
        )

    # Bloque de probabilidad
    if prob is None:
        prob_block = html.Div(
            html.P(
                f"El modelo {MODEL_LABELS_ES.get(model_name, model_name)} "
                "no entrega probabilidades en este caso, solo la clase predicha.",
                style={"fontSize": "0.85rem", "color": C_MUTED,
                       "fontStyle": "italic", "marginBottom": "0",
                       "textAlign": "center"},
            ),
            className="result-prob-block",
        )
    else:
        prob_pct = prob * 100
        bar_width = max(0, min(100, prob_pct))
        prob_block = html.Div(
            [
                html.P("Probabilidad estimada de clase positiva",
                       className="result-prob-label"),
                html.H2(
                    f"{prob_pct:.1f}%",
                    className="result-prob-value",
                    style={"color": prob_color},
                ),
                html.Div(
                    html.Div(
                        className="result-prob-bar-fill",
                        style={
                            "width": f"{bar_width}%",
                            "background": prob_bg_grad,
                        },
                    ),
                    className="result-prob-bar",
                ),
                html.Div(
                    [html.Span("0%"), html.Span("50%"), html.Span("100%")],
                    className="result-prob-scale",
                ),
            ],
            className="result-prob-block",
        )

    return html.Div(
        dbc.Card(
            [
                # Banner superior coloreado según resultado
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(banner_kicker,
                                       className="result-banner-kicker"),
                                html.H5(banner_title,
                                        className="result-banner-title"),
                            ]
                        ),
                    ],
                    className=banner_class,
                ),

                # Probabilidad grande + barra visual
                prob_block,

                # Cuerpo del panel
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(tag_text, className=tag_class),
                            ],
                            style={"marginBottom": "8px"},
                        ),

                        html.P("Modelo utilizado",
                               className="result-section-title"),
                        html.P(
                            MODEL_LABELS_ES.get(model_name, model_name),
                            style={"fontSize": "0.92rem", "fontWeight": "700",
                                   "color": C_NAVY, "marginBottom": "10px"},
                        ),

                        html.P("Interpretación",
                               className="result-section-title"),
                        dcc.Markdown(
                            message,
                            style={"fontSize": "0.86rem", "color": "#5D6D7E",
                                   "lineHeight": "1.75", "marginBottom": "0"},
                        ),

                        html.Div(
                            "Recordatorio: este resultado tiene fines "
                            "académicos y no reemplaza una valoración médica.",
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
# Mini badges de la matriz de confusión (TN / FP / FN / TP)
# ──────────────────────────────────────────────────────────────────────────────

def _build_cm_mini_cards(cm_data):
    """Construye las 4 mini tarjetas que acompañan a la matriz de confusión."""
    if not cm_data or "matrix" not in cm_data:
        return html.Div()

    matrix = cm_data["matrix"]
    # cm_data["matrix"] viene como lista de listas: [[TN, FP], [FN, TP]]
    try:
        tn = int(matrix[0][0])
        fp = int(matrix[0][1])
        fn = int(matrix[1][0])
        tp = int(matrix[1][1])
    except Exception:
        return html.Div()

    def _mini_card(tag, value, desc, tone):
        return html.Div(
            [
                html.Div(
                    html.Span(tag),
                    className="cm-mini-tag",
                ),
                html.P(f"{value:,}", className="cm-mini-value"),
                html.P(desc, className="cm-mini-desc"),
            ],
            className=f"cm-mini-card is-{tone}",
        )

    return [
        _mini_card("TN · Verdaderos negativos", tn,
                   "Personas sin diabetes correctamente clasificadas.",
                   tone="good"),
        _mini_card("FP · Falsos positivos", fp,
                   "Sanos marcados como positivos (alarma innecesaria).",
                   tone="warn"),
        _mini_card("FN · Falsos negativos", fn,
                   "Casos reales que el modelo NO detecta. Error crítico.",
                   tone="bad"),
        _mini_card("TP · Verdaderos positivos", tp,
                   "Casos con prediabetes/diabetes detectados.",
                   tone="good"),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Registro de callbacks
# ──────────────────────────────────────────────────────────────────────────────

# Lista de IDs del simulador (orden importa para mapear a column names)
SIM_INPUT_FIELDS = [
    "Sex", "Age", "Education", "Income",
    "HighBP", "HighChol", "CholCheck", "Stroke", "HeartDiseaseorAttack", "BMI",
    "Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
    "GenHlth", "DiffWalk", "AnyHealthcare", "NoDocbcCost",
    "MentHlth", "PhysHlth",
]


def register_model_callbacks(app):

    # ── 1) KPIs y métricas del modelo recomendado (estáticos, una sola vez) ──
    @app.callback(
        Output("rec-kpi-acc",       "children"),
        Output("rec-kpi-acc-sub",   "children"),
        Output("rec-kpi-prec",      "children"),
        Output("rec-kpi-prec-sub",  "children"),
        Output("rec-kpi-rec",       "children"),
        Output("rec-kpi-rec-sub",   "children"),
        Output("rec-kpi-f1",        "children"),
        Output("rec-kpi-f1-sub",    "children"),
        Output("rec-kpi-roc",       "children"),
        Output("rec-kpi-roc-sub",   "children"),
        Output("rec-kpi-pr",        "children"),
        Output("rec-kpi-pr-sub",    "children"),
        Output("recommended-model-name", "children"),
        Input("recommended-model-name", "id"),  # dispara al cargar el layout
    )
    def fill_recommended_kpis(_):
        df = get_metrics_df()
        if df.empty:
            dash = "—"
            return (dash, dash, dash, dash, dash, dash, dash, dash,
                    dash, dash, dash, dash, "Random Forest")
        best = get_best_model_name()
        row = df[df["model"] == best]
        if row.empty:
            row = df.iloc[[0]]
        r = row.iloc[0]
        acc  = fmt_pct(r["accuracy"])
        prec = fmt_pct(r["precision"])
        rec  = fmt_pct(r["recall"])
        f1   = fmt_pct(r["f1"])
        roc  = fmt_pct(r["roc_auc"])
        prx  = fmt_pct(r["pr_auc"])
        return (
            acc,  "Aciertos sobre el total",
            prec, "Confiabilidad de positivos",
            rec,  "Detección de la clase positiva",
            f1,   "Equilibrio precision/recall",
            roc,  "Capacidad de discriminación",
            prx,  "Desempeño en clase minoritaria",
            MODEL_LABELS_ES.get(best, best),
        )

    # ── 2) Selector de modelos: descripción + KPIs dinámicos ─────────────────
    @app.callback(
        Output("explorer-model-name",         "children"),
        Output("explorer-model-description",  "children"),
        Output("explorer-model-interpretation", "children"),
        Output("exp-kpi-acc",   "children"),
        Output("exp-kpi-prec",  "children"),
        Output("exp-kpi-rec",   "children"),
        Output("exp-kpi-f1",    "children"),
        Output("exp-kpi-roc",   "children"),
        Output("exp-kpi-pr",    "children"),
        Output("exp-kpi-acc-sub",  "children"),
        Output("exp-kpi-prec-sub", "children"),
        Output("exp-kpi-rec-sub",  "children"),
        Output("exp-kpi-f1-sub",   "children"),
        Output("exp-kpi-roc-sub",  "children"),
        Output("exp-kpi-pr-sub",   "children"),
        Input("model-explorer-select", "value"),
    )
    def update_model_explorer(model_name):
        df = get_metrics_df()
        if df.empty or model_name is None:
            dash = "—"
            return (dash, dash, dash, *([dash] * 6), *([""] * 6))

        row_match = df[df["model"] == model_name]
        if row_match.empty:
            row_match = df.iloc[[0]]
        r = row_match.iloc[0].to_dict()

        return (
            MODEL_LABELS_ES.get(model_name, model_name),
            MODEL_DESCRIPTIONS.get(model_name, ""),
            _interpretation_text(model_name, r),
            fmt_pct(r["accuracy"]),
            fmt_pct(r["precision"]),
            fmt_pct(r["recall"]),
            fmt_pct(r["f1"]),
            fmt_pct(r["roc_auc"]),
            fmt_pct(r["pr_auc"]),
            "Aciertos totales",
            "Confiabilidad +",
            "Detección de +",
            "Equilibrio P/R",
            "Discriminación",
            "Clase minoritaria",
        )

    # ── 3) Simulador: predicción al hacer clic ───────────────────────────────
    @app.callback(
        Output("sim-result-panel", "children"),
        Input("sim-predict-button", "n_clicks"),
        State("sim-model-select", "value"),
        *[State(f"sim-{name}", "value") for name in SIM_INPUT_FIELDS],
        prevent_initial_call=True,
    )
    def run_simulation(n_clicks, model_name, *values):
        if not n_clicks:
            return no_update

        form = dict(zip(SIM_INPUT_FIELDS, values))
        # Saneamiento básico: forzar BMI a número
        try:
            if form.get("BMI") is not None:
                form["BMI"] = float(form["BMI"])
        except Exception:
            form["BMI"] = 27.0

        label, prob = predict_with_model(model_name or get_best_model_name(),
                                         form)
        return _build_result_panel(model_name or get_best_model_name(),
                                   label, prob)

    # ── 4) Gráficas de evaluación del mejor modelo ───────────────────────────
    @app.callback(
        Output("ml-confusion-matrix", "figure"),
        Output("ml-roc-curve",        "figure"),
        Output("ml-pr-curve",         "figure"),
        Output("ml-feature-importance", "figure"),
        Input("ml-confusion-matrix", "id"),  # dummy: dispara al cargar
    )
    def render_evaluation_figures(_):
        cm = get_confusion_matrix()
        f_cm  = fig_confusion_matrix(cm)
        f_roc = fig_roc_curves()              # todas las curvas
        f_pr  = fig_pr_curves()
        f_imp = fig_feature_importance(top_n=12)
        return f_cm, f_roc, f_pr, f_imp

    # ── 5) Tabla comparativa de modelos ──────────────────────────────────────
    @app.callback(
        Output("ml-models-table", "children"),
        Input("ml-models-table", "id"),
    )
    def render_models_table(_):
        df = get_metrics_df()
        return _models_table(df)

    # ── 6) Gráfica comparativa con métricas seleccionables ───────────────────
    @app.callback(
        Output("ml-comparison-graph", "figure"),
        Input("ml-comp-metrics", "value"),
    )
    def render_comparison_graph(metrics):
        if not metrics:
            metrics = ["f1"]
        return fig_metrics_comparison(metrics)

    # ── 7) Botón "Limpiar formulario": restaura los 21 inputs a sus
    #     valores por defecto. NO interfiere con el botón "Predecir" porque
    #     usa Outputs distintos (los inputs en sí, no el panel de resultado).
    @app.callback(
        *[Output(f"sim-{name}", "value", allow_duplicate=True)
          for name in SIM_INPUT_FIELDS],
        Output("sim-model-select", "value", allow_duplicate=True),
        Input("sim-clear-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_simulator_form(n_clicks):
        if not n_clicks:
            return no_update
        defaults = [SIM_DEFAULTS[name] for name in SIM_INPUT_FIELDS]
        return (*defaults, get_best_model_name())

    # ── 8) Al pulsar "Limpiar", también restablecemos el panel de resultado.
    @app.callback(
        Output("sim-result-panel", "children", allow_duplicate=True),
        Input("sim-clear-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_result_panel_on_clear(n_clicks):
        if not n_clicks:
            return no_update
        return _initial_result_panel()
