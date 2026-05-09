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
    fig_class_imbalance, fig_confusion_matrix, fig_feature_importance,
    fig_metrics_comparison,
    fig_pr_curves, fig_roc_curves,
    # Nuevas figuras añadidas en el rediseño de la pestaña de Predicción
    fig_proba_distribution, fig_threshold_tradeoff, fig_three_metric_heatmaps,
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
def _interpretation_text(model_key, r):
    """
    Genera una interpretación cualitativa para el bloque azul del explorador
    de modelos. Evita repetir solo las métricas y explica el comportamiento
    de la combinación modelo + balanceo.
    """

    model_base = str(r.get("model_base", "")).strip()
    balance = str(r.get("balance_strategy", "")).strip()

    recall = float(r.get("recall", 0))
    precision = float(r.get("precision", 0))
    accuracy = float(r.get("accuracy", 0))
    f1 = float(r.get("f1", 0))

    balance_labels = {
        "none": "sin balanceo",
        "class_weight": "class weight",
        "smote": "SMOTE",
        "adasyn": "ADASYN",
    }

    balance_label = balance_labels.get(balance, balance)

    # Caso especial: modelo recomendado
    if model_base == "XGBoost" and balance == "class_weight":
        return (
            "Interpretación rápida: esta es la combinación recomendada porque prioriza la detección "
            "de la clase positiva. Su principal fortaleza es identificar una alta proporción de personas "
            "con prediabetes/diabetes, aunque esto implica aceptar una precisión moderada y una mayor cantidad "
            "de falsos positivos. Es adecuada cuando el objetivo principal es reducir los casos positivos no detectados."
        )

    # Modelos sin balanceo
    if balance == "none":
        if recall < 0.30 and accuracy >= 0.80:
            return (
                "Interpretación rápida: aunque la accuracy parece alta, este resultado puede ser engañoso "
                "por el desbalance del dataset. El modelo tiende a favorecer la clase mayoritaria y presenta "
                "baja capacidad para detectar correctamente los casos de prediabetes/diabetes."
            )

        return (
            "Interpretación rápida: esta configuración funciona como punto de referencia, ya que permite observar "
            "cómo se comporta el modelo sin aplicar correcciones frente al desbalance de clases. En general, puede "
            "favorecer la clase mayoritaria y perder sensibilidad frente a la clase positiva."
        )

    # Class weight
    if balance == "class_weight":
        if recall >= 0.70:
            return (
                f"Interpretación rápida: al usar {balance_label}, el modelo aumenta la atención sobre la clase "
                "minoritaria y mejora la detección de casos positivos. Esta configuración es útil cuando se busca "
                "priorizar el recall sin modificar directamente la distribución original de los datos."
            )

        return (
            f"Interpretación rápida: {balance_label} ajusta el peso de las clases durante el entrenamiento. "
            "Su objetivo es reducir el sesgo hacia la clase mayoritaria, aunque en esta combinación el aumento "
            "de sensibilidad frente a la clase positiva es más moderado."
        )

    # SMOTE
    if balance == "smote":
        if recall >= 0.55:
            return (
                "Interpretación rápida: SMOTE ayuda al modelo a reconocer mejor la clase positiva al generar "
                "ejemplos sintéticos de prediabetes/diabetes durante el entrenamiento. Esta estrategia puede mejorar "
                "la detección, aunque su desempeño depende de qué tan representativos sean los casos sintéticos creados."
            )

        return (
            "Interpretación rápida: SMOTE introduce ejemplos sintéticos de la clase minoritaria, pero en esta combinación "
            "no logra una mejora tan fuerte en la detección de positivos. Esto puede indicar que las clases siguen estando "
            "muy solapadas o que el modelo no aprovecha completamente el sobremuestreo."
        )

    # ADASYN
    if balance == "adasyn":
        if model_base == "KNN":
            return (
                "Interpretación rápida: ADASYN fue la estrategia más favorable para KNN bajo el criterio de recall. "
                "Al generar ejemplos sintéticos en las zonas más difíciles de clasificar, ayuda a que el modelo encuentre "
                "más vecinos representativos de la clase positiva. Aun así, KNN sigue siendo sensible a la distribución "
                "local de los datos y a la escala de las variables."
            )

        return (
            "Interpretación rápida: ADASYN concentra el sobremuestreo en los casos positivos más difíciles de aprender. "
            "Esto puede mejorar la sensibilidad del modelo, pero también puede aumentar los falsos positivos si las zonas "
            "difíciles están muy mezcladas con la clase mayoritaria."
        )

    # Fallback general según patrón de métricas
    if recall >= 0.70 and precision < 0.40:
        return (
            "Interpretación rápida: el modelo presenta alta sensibilidad para detectar casos positivos, aunque con precisión "
            "moderada. Esto indica que identifica muchas personas en riesgo, pero también puede generar más alertas falsas."
        )

    if recall < 0.35 and accuracy >= 0.80:
        return (
            "Interpretación rápida: el modelo obtiene buenos aciertos globales, pero detecta pocos casos positivos. "
            "Este comportamiento es típico en datasets desbalanceados cuando el modelo favorece la clase mayoritaria."
        )

    if f1 >= 0.45:
        return (
            "Interpretación rápida: esta combinación ofrece un equilibrio razonable entre detección de positivos y control "
            "de falsos positivos. Puede considerarse una alternativa intermedia cuando no se quiere priorizar únicamente recall."
        )

    return (
        "Interpretación rápida: esta configuración permite comparar cómo cambia el comportamiento del modelo bajo una técnica "
        "específica de balanceo. Su utilidad principal está en observar el intercambio entre recall, precisión y desempeño global."
    )


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
                                html.Li("Revisa los valores del perfil."),
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
    Output("rec-kpi-acc", "children"),
    Output("rec-kpi-acc-sub", "children"),
    Output("rec-kpi-prec", "children"),
    Output("rec-kpi-prec-sub", "children"),
    Output("rec-kpi-rec", "children"),
    Output("rec-kpi-rec-sub", "children"),
    Output("rec-kpi-f1", "children"),
    Output("rec-kpi-f1-sub", "children"),
    Output("rec-kpi-roc", "children"),
    Output("rec-kpi-roc-sub", "children"),
    Output("rec-kpi-pr", "children"),
    Output("rec-kpi-pr-sub", "children"),
    Output("recommended-model-name", "children"),
    Output("recommended-balance-technique", "children"),
    Output("recommended-model-context", "children"),
    Output("recommended-criterion", "children"),
    Output("recommended-dataset-note", "children"),
    Input("recommended-model-name", "id"),
    )
    
    def fill_recommended_kpis(_):
        df = get_metrics_df()

        if df.empty:
            dash = "—"
            return (
                dash, dash,
                dash, dash,
                dash, dash,
                dash, dash,
                dash, dash,
                dash, dash,
                "XGBoost",
                "Técnica de balanceo: —",
                "No se encontraron métricas disponibles para construir la recomendación.",
                "Criterio principal: Recall",
                "Dataset desbalanceado",
            )

        best = get_best_model_name()
        row = df[df["model"] == best]
        if row.empty:
            row = df.iloc[[0]]

        r = row.iloc[0]

        balance_labels = {
            "none": "Sin balanceo",
            "class_weight": "Class weight",
            "smote": "SMOTE",
            "adasyn": "ADASYN",
        }

        base_model = str(r.get("model_base", best.split("+")[0].strip()))
        balance_strategy = str(
            r.get(
                "balance_strategy",
                best.split("+")[1].strip() if "+" in best else "none"
            )
        )
        balance_label = balance_labels.get(balance_strategy, balance_strategy)

        acc = fmt_pct(r["accuracy"])
        prec = fmt_pct(r["precision"])
        rec = fmt_pct(r["recall"])
        f1 = fmt_pct(r["f1"])
        roc = fmt_pct(r["roc_auc"])
        prx = fmt_pct(r["pr_auc"])

        context = (
            f"{base_model} fue seleccionado como modelo recomendado porque obtuvo el mayor recall "
            f"en el conjunto de prueba ({rec}), criterio priorizado en este proyecto para maximizar "
            f"la detección de personas con prediabetes/diabetes. "
            f"Se utiliza la técnica de balanceo {balance_label}, adecuada para un dataset desbalanceado. "
            f"Aunque la precision ({prec}) es más moderada, esta combinación resulta conveniente cuando "
            f"el costo de no detectar casos positivos es alto."
        )

        return (
            acc, "Aciertos sobre el total",
            prec, "Confiabilidad de positivos",
            rec, "Detención de la clase positiva",
            f1, "Equilibrio entre precision y recall",
            roc, "Capacidad de discriminación",
            prx, "Desempeño en clase minoritaria",
            base_model,
            f"Técnica de balanceo: {balance_label}",
            context,
            "Criterio principal: Recall",
            "Dataset desbalanceado",
        )

    
    # ── 2A) Opciones de balanceo según el modelo base ───────────────────────
    @app.callback(
        Output("balance-technique-radio", "options"),
        Output("balance-technique-radio", "value"),
        Input("model-base-dropdown", "value"),
    )
    def update_balance_options(selected_model):
        options_map = {
            "Regresión Logística": [
                {"label": "Sin balanceo", "value": "none"},
                {"label": "Class weight", "value": "class_weight"},
                {"label": "SMOTE", "value": "smote"},
                {"label": "ADASYN", "value": "adasyn"},
            ],
            "Random Forest": [
                {"label": "Sin balanceo", "value": "none"},
                {"label": "Class weight", "value": "class_weight"},
                {"label": "SMOTE", "value": "smote"},
                {"label": "ADASYN", "value": "adasyn"},
            ],
            "XGBoost": [
                {"label": "Sin balanceo", "value": "none"},
                {"label": "Class weight", "value": "class_weight"},
                {"label": "SMOTE", "value": "smote"},
                {"label": "ADASYN", "value": "adasyn"},
            ],
            "KNN": [
                {"label": "Sin balanceo", "value": "none"},
                {"label": "SMOTE", "value": "smote"},
                {"label": "ADASYN", "value": "adasyn"},
            ],
        }

        default_map = {
            "Regresión Logística": "class_weight",
            "Random Forest": "class_weight",
            "XGBoost": "class_weight",
            "KNN": "adasyn",
        }

        return options_map.get(selected_model, []), default_map.get(selected_model)
    
     # ── 2B) Selector de modelos: descripción + KPIs dinámicos ────────────────
    @app.callback(
        Output("selected-combination-summary", "children"),
        Output("explorer-model-name", "children"),
        Output("explorer-model-description", "children"),
        Output("explorer-model-interpretation", "children"),
        Output("exp-kpi-acc", "children"),
        Output("exp-kpi-prec", "children"),
        Output("exp-kpi-rec", "children"),
        Output("exp-kpi-f1", "children"),
        Output("exp-kpi-roc", "children"),
        Output("exp-kpi-pr", "children"),
        Output("exp-kpi-acc-sub", "children"),
        Output("exp-kpi-prec-sub", "children"),
        Output("exp-kpi-rec-sub", "children"),
        Output("exp-kpi-f1-sub", "children"),
        Output("exp-kpi-roc-sub", "children"),
        Output("exp-kpi-pr-sub", "children"),
        Input("model-base-dropdown", "value"),
        Input("balance-technique-radio", "value"),
    )
    def update_model_explorer(model_base, balanceo):
        df = get_metrics_df()

        dash = "—"

        if df.empty or model_base is None or balanceo is None:
            return (
                dbc.Alert("No hay información disponible para esta combinación.", color="warning"),
                dash, dash, dash,
                *([dash] * 6),
                *([""] * 6),
            )

        row_match = df[
            (df["model_base"] == model_base) &
            (df["balance_strategy"] == balanceo)
        ]

        if row_match.empty:
            return (
                dbc.Alert("No se encontró información para esta combinación.", color="warning"),
                model_base,
                "No hay descripción disponible para esta combinación.",
                "Selecciona otra técnica de balanceo para revisar sus métricas.",
                *([dash] * 6),
                *([""] * 6),
            )

        r = row_match.iloc[0].to_dict()

        balance_labels = {
            "none": "Sin balanceo",
            "class_weight": "Class weight",
            "smote": "SMOTE",
            "adasyn": "ADASYN",
        }

        balance_label = balance_labels.get(balanceo, balanceo)
        model_key = r.get("model", f"{model_base} + {balanceo}")

        summary = html.Div(
            [
                html.Div("Configuración seleccionada", className="summary-title"),
                html.Div(
                    [
                        html.Span("Modelo base: ", className="summary-label"),
                        html.Span(model_base, className="summary-value"),
                    ]
                ),
                html.Div(
                    [
                        html.Span("Técnica de balanceo: ", className="summary-label"),
                        html.Span(balance_label, className="summary-value"),
                    ]
                ),
                html.Div(
                    [
                        html.Span("Recall en test: ", className="summary-label"),
                        html.Span(fmt_pct(r["recall"]), className="summary-value emphasis"),
                    ]
                ),
            ],
            className="summary-box",
        )

        return (
            summary,
            f"{model_base} + {balance_label}",
            MODEL_DESCRIPTIONS.get(
                model_key,
                MODEL_DESCRIPTIONS.get(
                    f"{model_base} + {balanceo}",
                    MODEL_DESCRIPTIONS.get(
                        f"{model_base} + {balance_label}",
                        MODEL_DESCRIPTIONS.get(model_base, "")
                    )
                )
            ),
            _interpretation_text(model_key, r),
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
    
    def _prediction_result_panel(pred_class, pred_prob):
        """Panel visual para mostrar el resultado de la simulación."""

        is_positive = int(pred_class) == 1

        if pred_prob is None:
            prob_text = "No disponible"
            prob_pct = None
        else:
            prob_pct = float(pred_prob) * 100
            prob_text = f"{prob_pct:.2f}%"

        if is_positive:
            title = "Prediabetes / Diabetes"
            subtitle = "El modelo clasifica este perfil dentro del grupo positivo."
            badge = "Riesgo estimado positivo"
            accent = "#C0392B"
            bg = "linear-gradient(180deg, #FFF7F6 0%, #FFFFFF 100%)"
            icon = "fa-solid fa-triangle-exclamation"
        else:
            title = "Sin diabetes"
            subtitle = "El modelo clasifica este perfil dentro del grupo sin diabetes."
            badge = "Riesgo estimado bajo"
            accent = "#1E8449"
            bg = "linear-gradient(180deg, #F4FFF8 0%, #FFFFFF 100%)"
            icon = "fa-solid fa-circle-check"

        return dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.Span(
                                [
                                    html.I(className=icon, style={"marginRight": "8px"}),
                                    badge,
                                ],
                                style={
                                    "display": "inline-flex",
                                    "alignItems": "center",
                                    "padding": "7px 12px",
                                    "borderRadius": "999px",
                                    "backgroundColor": "#FFFFFF",
                                    "border": f"1px solid {accent}",
                                    "color": accent,
                                    "fontSize": "0.78rem",
                                    "fontWeight": "800",
                                    "letterSpacing": "0.04em",
                                    "textTransform": "uppercase",
                                    "marginBottom": "14px",
                                },
                            ),
                        ]
                    ),

                    html.H3(
                        title,
                        style={
                            "fontSize": "2rem",
                            "fontWeight": "900",
                            "color": "#0F2447",
                            "marginBottom": "8px",
                        },
                    ),

                    html.P(
                        subtitle,
                        style={
                            "fontSize": "0.95rem",
                            "color": "#5D6D7E",
                            "lineHeight": "1.7",
                            "marginBottom": "18px",
                        },
                    ),

                    html.Div(
                        [
                            html.Div(
                                "Probabilidad estimada",
                                style={
                                    "fontSize": "0.78rem",
                                    "fontWeight": "800",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.06em",
                                    "color": "#5D6D7E",
                                    "marginBottom": "4px",
                                },
                            ),
                            html.Div(
                                prob_text,
                                style={
                                    "fontSize": "2.4rem",
                                    "fontWeight": "900",
                                    "color": accent,
                                    "lineHeight": "1.1",
                                },
                            ),
                        ],
                        style={
                            "backgroundColor": "#F8FBFF",
                            "border": "1px solid #DDE7F2",
                            "borderRadius": "16px",
                            "padding": "16px 18px",
                            "marginBottom": "16px",
                        },
                    ),

                    html.Div(
                        [
                            html.I(
                                className="fa-solid fa-circle-info",
                                style={"marginRight": "8px", "color": "#4A7FC1"},
                            ),
                            html.Span(
                                "Este resultado es una estimación académica generada por el modelo final "
                                "XGBoost + Class weight. No reemplaza una valoración médica.",
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "flex-start",
                            "fontSize": "0.84rem",
                            "lineHeight": "1.6",
                            "color": "#5D6D7E",
                            "backgroundColor": "#EEF5FC",
                            "borderLeft": "4px solid #4A7FC1",
                            "borderRadius": "10px",
                            "padding": "12px 14px",
                        },
                    ),
                ],
                className="p-4",
            ),
            className="border-0 shadow-sm",
            style={
                "borderRadius": "18px",
                "background": bg,
                "borderTop": f"4px solid {accent}",
                "marginTop": "18px",
            },
        )


    def _error_result_panel(message):
        """Panel de error para el simulador."""
        return dbc.Alert(
            [
                html.Strong("No fue posible calcular la predicción. "),
                html.Span(message),
            ],
            color="danger",
            style={
                "borderRadius": "12px",
                "fontSize": "0.9rem",
                "lineHeight": "1.6",
            },
        )
    
    
    
    
    @app.callback(
        Output("sim-result-panel", "children"),
        Input("sim-predict-button", "n_clicks"),
        *[State(f"sim-{name}", "value") for name in SIM_INPUT_FIELDS],
        prevent_initial_call=True,
    )
    def run_simulation(n_clicks, *values):
        if not n_clicks:
            return no_update

        # Modelo fijo recomendado para la simulación
        selected_model = "Mejor modelo"

        form_values = dict(zip(SIM_INPUT_FIELDS, values))

        pred_class, pred_prob = predict_with_model(selected_model, form_values)

        if pred_class is None:
            return _error_result_panel(
                "No fue posible calcular la predicción. Verifica que el modelo final esté disponible en la carpeta models/."
            )

        return _prediction_result_panel(pred_class, pred_prob)
    # ── 4) Gráficas de evaluación del mejor modelo ───────────────────────────
    @app.callback(
        Output("ml-class-imbalance", "figure"),
        Output("ml-confusion-matrix", "figure"),
        Output("ml-feature-importance", "figure"),
        Input("ml-confusion-matrix", "id"),  # dummy: dispara al cargar
    )
    def render_evaluation_figures(_):
        f_imb = fig_class_imbalance()
        f_cm  = fig_confusion_matrix()
        f_imp = fig_feature_importance(top_n=12)
        return f_imb, f_cm, f_imp

    @app.callback(
        Output("ml-roc-curve", "figure"),
        Output("ml-pr-curve", "figure"),
        Output("ml-curves-note", "children"),
        Input("ml-curve-model-selector", "value"),
    )
    def render_curve_figures(selected_models):
        note = ""
        if isinstance(selected_models, str):
            selected_models = [selected_models]
        if not selected_models:
            selected_models = None
            note = "No hay modelos seleccionados; se muestran todas las curvas disponibles."
        elif len(selected_models) == 1:
            note = "Solo hay una curva seleccionada para la comparación."
        else:
            note = f"{len(selected_models)} curvas seleccionadas para comparar modelos base."
        return (
            fig_roc_curves(selected_models=selected_models),
            fig_pr_curves(selected_models=selected_models),
            note,
        )

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
            metrics = ["recall"]
        return fig_metrics_comparison(metrics)

    # ── 7) Botón "Limpiar formulario": restaura los 21 inputs a sus
    #     valores por defecto. NO interfiere con el botón "Predecir" porque
    #     usa Outputs distintos (los inputs en sí, no el panel de resultado).
    @app.callback(
        *[Output(f"sim-{name}", "value", allow_duplicate=True)
          for name in SIM_INPUT_FIELDS],
        Input("sim-clear-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_simulator_form(n_clicks):
        if not n_clicks:
            return no_update
        defaults = [SIM_DEFAULTS[name] for name in SIM_INPUT_FIELDS]
        return tuple(defaults)

    # ── 8) Al pulsar "Limpiar", también restablecemos el panel de resultado.
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

    # ── 9) Distribución de probabilidades predichas (mejor modelo) ───────────
    #     Histograma por clase real con línea vertical en el threshold final.
    @app.callback(
        Output("ml-proba-distribution", "figure"),
        Input("ml-proba-distribution", "id"),  # dummy: se dispara al cargar
    )
    def render_proba_distribution(_):
        return fig_proba_distribution()

    # ── 10) Trade-off Precision/Recall/F1 vs threshold ───────────────────────
    @app.callback(
        Output("ml-threshold-tradeoff", "figure"),
        Input("ml-threshold-tradeoff", "id"),
    )
    def render_threshold_tradeoff(_):
        return fig_threshold_tradeoff()

    # ── 11) Heatmap Recall (Modelo_base × Balanceo) ──────────────────────────
    @app.callback(
        Output("ml-three-metric-heatmaps", "figure"),
        Input("ml-three-metric-heatmaps", "id"),
    )
    def render_three_metric_heatmaps(_):
        return fig_three_metric_heatmaps()
