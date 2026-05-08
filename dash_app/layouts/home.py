from dash import html
import dash_bootstrap_components as dbc

C_NAVY = "#0F2447"
C_BLUE = "#1B3A6B"
C_RED = "#C0392B"
C_BLUEI = "#4A7FC1"
C_BG = "#F4F6F9"
C_TEXT = "#2C3E50"
C_MUTED = "#7F8C8D"
C_BORDER = "#DDE2E8"
C_CARD = "#FFFFFF"
C_SOFT = "#EAF1FB"
C_GREEN = "#1E8449"
C_GOLD = "#D4AC0D"


def layout_home():
    return dbc.Container(
        [
            # HERO PRINCIPAL
            # HERO PRINCIPAL CON IMAGEN
            html.Div([

                # overlay oscuro (CLAVE)
                html.Div(style={
                    "position": "absolute",
                    "top": 0,
                    "left": 0,
                    "width": "100%",
                    "height": "100%",
                    "background": "rgba(15,36,71,0.65)",
                    "zIndex": 1
                }),

                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            "BRFSS 2015 · CDC · Dashboard Analítico",
                                            style={
                                                "display": "inline-block",
                                                "backgroundColor": "rgba(255,255,255,0.2)",  # cambio
                                                "color": "#FFFFFF",
                                                "padding": "8px 16px",
                                                "borderRadius": "999px",
                                                "fontSize": "0.76rem",
                                                "fontWeight": "700",
                                                "letterSpacing": "0.05em",
                                                "marginTop": "48px",
                                                "marginBottom": "20px",
                                            },
                                        ),
                                        html.H1(
                                            "¿Qué factores están más asociados con la diabetes en esta población?",
                                            style={
                                                "fontWeight": "800",
                                                "fontSize": "3.8rem",
                                                "lineHeight": "1.1",
                                                "color": "#FFFFFF",  #cambio
                                                "marginBottom": "20px",
                                                "maxWidth": "850px",
                                            },
                                        ),
                                        html.P(
                                            "Explora patrones clínicos, conductuales y sociodemográficos "
                                            "relacionados con diabetes y prediabetes a través de una interfaz "
                                            "clara, visual e interactiva.",
                                            style={
                                                "fontSize": "1rem",
                                                "color": "rgba(255,255,255,0.85)",  # cambio
                                                "lineHeight": "1.85",
                                                "maxWidth": "720px",
                                                "marginBottom": "28px",
                                            },
                                        ),

                                        #dbc.Row([...], className="g-2"),  
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Button(
                                                        "Entrar al dashboard",
                                                        href="/dashboard",
                                                        style={
                                                            "backgroundColor": C_BLUEI,
                                                            "borderColor": C_BLUEI,
                                                            "fontWeight": "700",
                                                            "fontSize": "0.95rem",
                                                            "padding": "12px 24px",
                                                            "borderRadius": "10px",
                                                            "boxShadow": "0 8px 18px rgba(27,58,107,0.14)",
                                                        },
                                                    ),
                                                    xs=12,
                                                    sm="auto",
                                                ),
                                                dbc.Col(
                                                    dbc.Button(
                                                        "Ver análisis exploratorio",
                                                        href="/about",
                                                        outline=True,
                                                        style={
                                                            "fontWeight": "700",
                                                            "fontSize": "0.95rem",
                                                            "padding": "12px 24px",
                                                            "borderRadius": "10px",
                                                            "color": C_TEXT,
                                                            "borderColor": C_BORDER,
                                                            "backgroundColor": "#FFFFFF",
                                                        },
                                                    ),
                                                    xs=12,
                                                    sm="auto",
                                                ),
                                            ],
                                            className="g-2",
                                        ),
        
                                        
                                        
                                        # deja tus botones igual
                                    ]
                                )
                            ],
                            lg=8,
                        ),

                        dbc.Col(
                            [
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.P("Vista rápida", style=_kicker()),
                                            html.H4(
                                                "Panorama del proyecto",
                                                style={
                                                    "fontWeight": "700",
                                                    "color": "#FFFFFF",
                                                    "fontSize": "1.15rem",
                                                    "marginBottom": "16px",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    _mini_stat("253.680", "registros"),
                                                    _mini_stat("22", "variables"),
                                                    _mini_stat("13,9%", "prevalencia"),
                                                ],
                                                style={
                                                    "display": "grid",
                                                    "gridTemplateColumns": "1fr",
                                                    "gap": "12px",
                                                },
                                            ),
                                        ],
                                        className="p-4",
                                    ),
                                    className="border-0",
                                    style={
                                        "borderRadius": "18px",
                                        "background": "rgba(255,255,255,0.15)",  # glass effect
                                        "backdropFilter": "blur(8px)",
                                        "border": "1px solid rgba(255,255,255,0.2)",
                                        "marginTop": "56px",
                                    },
                                )
                            ],
                            lg=4,
                        ),
                    ],
                    className="mb-5 align-items-start",
                    style={"position": "relative", "zIndex": 2}
                ),

            ],
            style={
                "backgroundImage": "url('/assets/fondo (2).png')",
                "backgroundSize": "cover",
                "backgroundPosition": "center",
                "backgroundRepeat": "no-repeat",
                "padding": "18px 40px",
                "borderRadius": "18px",
                "marginTop": "20px",
                "marginBottom": "25px",
                "position": "relative",
                "overflow": "hidden",
                "boxShadow": "0 10px 30px rgba(0,0,0,0.08)"
            }),
                        
            
            
            
            
            # BLOQUE DE PREGUNTAS ANALÍTICAS
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.P("Preguntas que guía este proyecto", style=_kicker()),
                                    html.H3(
                                        "¿Qué puedes descubrir aquí?",
                                        style={
                                            "fontWeight": "800",
                                            "color": C_NAVY,
                                            "fontSize": "2.6rem",
                                            "marginBottom": "20px",
                                        },
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                _question_box(
                                                    "¿Qué variables diferencian más claramente a las personas con diabetes?"
                                                ),
                                                md=4,
                                            ),
                                            dbc.Col(
                                                _question_box(
                                                    "¿Cómo cambian los patrones según edad, salud percibida y condiciones clínicas?"
                                                ),
                                                md=4,
                                            ),
                                            dbc.Col(
                                                _question_box(
                                                    "¿Qué señales pueden ayudar a construir perfiles de mayor riesgo?"
                                                ),
                                                md=4,
                                            ),
                                        ],
                                        className="g-3",
                                    ),
                                ],
                                className="p-4 p-md-5",
                            ),
                            className="border-0 shadow-sm",
                            style={"borderRadius": "18px", "backgroundColor": "#FFFFFF"},
                        )
                    )
                ],
                className="mb-5",
            ),

            # INSIGHTS
            html.Div(id="insights-home"),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.P("Resumen ejecutivo", style=_kicker(center=True)),
                                html.H3(
                                    "Hallazgos que orientan la exploración",
                                    style={
                                        "fontWeight": "800",
                                        "color": C_NAVY,
                                        "fontSize": "1.9rem",
                                        "marginBottom": "12px",
                                        "textAlign": "center",
                                    },
                                ),
                                html.P(
                                    "Antes de entrar al dashboard, estas son las ideas clave que contextualizan "
                                    "el análisis y ayudan a interpretar mejor los patrones observados.",
                                    style={
                                        "fontSize": "0.96rem",
                                        "color": "#5D6D7E",
                                        "lineHeight": "1.8",
                                        "textAlign": "center",
                                        "maxWidth": "760px",
                                        "margin": "0 auto 28px auto",
                                    },
                                ),
                            ]
                        ),
                        width=12,
                    )
                ],
                className="mb-2",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        _insight_card(
                            title="La diabetes no se distribuye de forma aleatoria",
                            body=(
                                "Los patrones observados sugieren una mayor presencia de diabetes en perfiles "
                                "con señales clínicas y de salud general menos favorables."
                            ),
                            color=C_RED,
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        _insight_card(
                            title="El riesgo parece responder a múltiples dimensiones",
                            body=(
                                "La lectura analítica gana valor cuando se integran factores clínicos, hábitos "
                                "de vida y contexto sociodemográfico en conjunto."
                            ),
                            color=C_BLUE,
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        _insight_card(
                            title="El dashboard funciona como herramienta de interpretación",
                            body=(
                                "Más que mostrar gráficos, la interfaz ayuda a recorrer el análisis de forma "
                                "clara, visual y orientada a comprensión."
                            ),
                            color=C_GOLD,
                        ),
                        md=4,
                    ),
                ],
                className="g-3 mb-5",
            ),

            # CIERRE / GUÍA
            dbc.Row(
                [
                    dbc.Col(
                        _bottom_card(
                            "¿Por dónde empezar?",
                            "Puedes ir directamente al dashboard para explorar patrones visuales o revisar primero el análisis exploratorio para entender el contexto, las variables y la lógica del estudio.",
                            C_BLUE,
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        _bottom_card(
                            "¿Para qué sirve esta interfaz?",
                            "Esta pantalla de inicio sintetiza el trabajo realizado y te orienta antes de navegar el dashboard, ayudando a interpretar mejor los hallazgos y el alcance del proyecto.",
                            C_GREEN,
                        ),
                        md=6,
                    ),
                ],
                className="g-3 pb-5",
            ),
        ],
        fluid=True,
        className="px-4",
        style={"backgroundColor": C_BG, "minHeight": "100vh"},
    )


def _kicker(center=False):
    return {
        "fontSize": "0.74rem",
        "fontWeight": "700",
        "textTransform": "uppercase",
        "letterSpacing": "0.08em",
        "color": C_MUTED,
        "marginBottom": "8px",
        "textAlign": "center" if center else "left",
    }


def _mini_stat(value, label):
    return html.Div(
        [
            html.H4(
                value,
                style={
                    "fontWeight": "800",
                    "fontSize": "1.25rem",
                    "color": C_NAVY,
                    "marginBottom": "3px",
                },
            ),
            html.P(
                label,
                style={
                    "fontSize": "0.73rem",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                    "fontWeight": "700",
                    "color": C_MUTED,
                    "marginBottom": "0",
                },
            ),
        ],
        className="mini-stat-box",
        style={
            "backgroundColor": "#FFFFFF",
            "border": f"1px solid {C_BORDER}",
            "borderRadius": "12px",
            "padding": "14px 14px",
            "textAlign": "center",
        },
    )


def _path_card(title, subtitle, body, button_text, href, color):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    style={
                        "width": "36px",
                        "height": "4px",
                        "backgroundColor": color,
                        "borderRadius": "3px",
                        "marginBottom": "14px",
                    }
                ),
                html.P(
                    subtitle,
                    style={
                        "fontSize": "0.72rem",
                        "fontWeight": "700",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.06em",
                        "color": C_MUTED,
                        "marginBottom": "8px",
                    },
                ),
                html.H5(
                    title,
                    style={
                        "fontWeight": "800",
                        "color": C_TEXT,
                        "fontSize": "1rem",
                        "marginBottom": "10px",
                    },
                ),
                html.P(
                    body,
                    style={
                        "fontSize": "0.88rem",
                        "color": "#5D6D7E",
                        "lineHeight": "1.8",
                        "marginBottom": "18px",
                    },
                ),
                dbc.Button(
                    button_text,
                    href=href,
                    outline=True if href == "#insights-home" else False,
                    style={
                        "backgroundColor": color if href != "#insights-home" else "#FFFFFF",
                        "borderColor": color,
                        "color": "#FFFFFF" if href != "#insights-home" else color,
                        "fontWeight": "700",
                        "borderRadius": "8px",
                        "padding": "10px 18px",
                    },
                ),
            ],
            className="p-4",
        ),
        className="border-0 shadow-sm h-100",
        style={"borderRadius": "16px", "backgroundColor": "#FFFFFF"},
    )


def _question_box(text):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    style={
                        "width": "30px",
                        "height": "4px",
                        "backgroundColor": C_BLUEI,
                        "borderRadius": "3px",
                        "marginBottom": "14px",
                    }
                ),
                html.P(
                    text,
                    style={
                        "fontSize": "1.0rem",
                        "fontWeight": "600",
                        "color": C_TEXT,
                        "lineHeight": "1.75",
                        "marginBottom": "0",
                    },
                ),
            ],
            className="p-4",
        ),
        className="border-0 h-100",
        style={
            "borderRadius": "14px",
            "backgroundColor": "#ECEDEE",
            "border": f"1px solid {C_BORDER}",
        },
    )


def _insight_card(title, body, color):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    style={
                        "width": "44px",
                        "height": "4px",
                        "backgroundColor": color,
                        "borderRadius": "3px",
                        "marginBottom": "14px",
                    }
                ),
                html.H6(
                    title,
                    style={
                        "fontWeight": "800",
                        "fontSize": "1.1rem",
                        "color": C_TEXT,
                        "marginBottom": "10px",
                        "lineHeight": "1.5",
                    },
                ),
                html.P(
                    body,
                    style={
                        "fontSize": "1.0rem",
                        "color": "#5D6D7E",
                        "lineHeight": "1.8",
                        "marginBottom": "0",
                    },
                ),
            ],
            className="p-4",
        ),
        className="border-0 shadow-sm h-100",
        style={"borderRadius": "16px", "backgroundColor": "#FFFFFF"},
    )


def _metric_tile(title, value):
    return dbc.Card(
        dbc.CardBody(
            [
                html.P(
                    title,
                    style={
                        "fontSize": "0.73rem",
                        "fontWeight": "700",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.05em",
                        "color": C_MUTED,
                        "marginBottom": "8px",
                    },
                ),
                html.H5(
                    value,
                    style={
                        "fontWeight": "800",
                        "fontSize": "1rem",
                        "color": C_NAVY,
                        "marginBottom": "0",
                        "lineHeight": "1.5",
                    },
                ),
            ],
            className="p-3",
        ),
        className="border-0 h-100",
        style={
            "borderRadius": "14px",
            "backgroundColor": "#FAFBFC",
            "border": f"1px solid {C_BORDER}",
        },
    )


def _bottom_card(title, body, color):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    style={
                        "width": "34px",
                        "height": "4px",
                        "backgroundColor": color,
                        "borderRadius": "3px",
                        "marginBottom": "14px",
                    }
                ),
                html.H6(
                    title,
                    style={
                        "fontWeight": "800",
                        "fontSize": "1.1rem",
                        "color": C_TEXT,
                        "marginBottom": "10px",
                    },
                ),
                html.P(
                    body,
                    style={
                        "fontSize": "1.0rem",
                        "color": "#5D6D7E",
                        "lineHeight": "1.8",
                        "marginBottom": "0",
                    },
                ),
            ],
            className="p-4",
        ),
        className="border-0 shadow-sm h-100",
        style={"borderRadius": "16px", "backgroundColor": "#FFFFFF"},
    )
