from dash import html
import dash_bootstrap_components as dbc


C_NAVY = "#0F2447"
C_BLUE = "#1B3A6B"
C_BLUEI = "#4A7FC1"
C_BG = "#F4F6F9"
C_TEXT = "#2C3E50"
C_MUTED = "#7F8C8D"
C_SOFT = "#EAF1FB"


def layout_team():
    return dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            "Equipoooooo del proyecto",
                                            style={
                                                "fontWeight": "700",
                                                "fontSize": "0.95rem",
                                                "color": C_BLUE,
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "inline-flex",
                                        "alignItems": "center",
                                        "backgroundColor": C_SOFT,
                                        "padding": "8px 14px",
                                        "borderRadius": "999px",
                                        "marginBottom": "16px",
                                    },
                                ),
                                html.H2(
                                    "Quiénes desarrollaron este proyecto",
                                    style={
                                        "fontWeight": "900",
                                        "color": C_NAVY,
                                        "marginBottom": "10px",
                                        "fontSize": "2rem",
                                    },
                                ),
                                html.P(
                                    "Esta sección presenta a las integrantes del proyecto y su aporte dentro del análisis, "
                                    "la visualización y la interpretación de los resultados.",
                                    style={
                                        "fontSize": "0.98rem",
                                        "color": C_MUTED,
                                        "maxWidth": "920px",
                                        "marginBottom": "0",
                                        "lineHeight": "1.8",
                                    },
                                ),
                            ]
                        ),
                        className="border-0 shadow-sm",
                        style={"borderRadius": "18px", "backgroundColor": "#FFFFFF"},
                    ),
                    width=12,
                ),
                className="pt-4 pb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        _team_card(
                            image_path="/assets/Daniela.jpeg",
                            initials="DH",
                            name="Daniela Hernández Navas",
                            role="Estudiante de Ciencia de Datos",
                            description=(
                                "Estudiante de sexto semestre de Ciencia de Datos, apasionada por entender los datos "
                                "y transformarlos en soluciones útiles. Le interesan especialmente el análisis de datos, "
                                "la inteligencia artificial, la automatización y la visualización, siempre buscando "
                                "aplicar lo que aprende en proyectos reales. Se caracteriza por su curiosidad, su "
                                "pensamiento crítico, su enfoque práctico y sus ganas constantes de seguir aprendiendo."
                            ),
                            contribution=(
                                "Participó en la estructuración analítica del proyecto, el diseño de visualizaciones "
                                "y la interpretación de patrones relacionados con diabetes y prediabetes."
                            ),
                            tags=["Machine Learning", "Data Analysis", "Statistics"],
                            linkedin_url="https://www.linkedin.com/in/daniela-hernández-a32572264?utm_source=share_via&utm_content=profile&utm_medium=member_ios",
                            accent_color=C_BLUE,
                        ),
                        md=6,
                        className="mb-4",
                    ),
                    dbc.Col(
                        _team_card(
                            image_path="/assets/Valeria.jpeg",
                            initials="VI",
                            name="Valeria Incer Vergara",
                            role="Estudiante de Ciencia de Datos y Matemáticas",
                            description=(
                                "Estudiante de sexto semestre de Ciencia de Datos y de Matemáticas, con un fuerte "
                                "interés en el uso de los datos para comprender y resolver problemas reales. Disfruta "
                                "trabajar con análisis de datos y modelos de aprendizaje automático, integrando su "
                                "formación matemática en cada proyecto. Se destaca por su pensamiento lógico, su "
                                "curiosidad y su compromiso con seguir aprendiendo y creciendo en el campo."
                            ),
                            contribution=(
                                "Contribuyó en la exploración del dataset, la organización metodológica y la "
                                "construcción de una lectura clara y estructurada de los hallazgos."
                            ),
                            tags=["EDA", "Mathematics", "Research"],
                            linkedin_url=None,
                            accent_color=C_BLUEI,
                        ),
                        md=6,
                        className="mb-4",
                    ),
                ],
                className="g-4",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5(
                                    "Sobre el proyecto",
                                    style={
                                        "fontWeight": "800",
                                        "color": C_NAVY,
                                        "marginBottom": "10px",
                                    },
                                ),
                                html.P(
                                    "Este dashboard fue desarrollado como una interfaz visual para comunicar de forma "
                                    "más clara los resultados del análisis exploratorio del dataset BRFSS 2015, "
                                    "integrando contexto, metodología, hallazgos y componentes interactivos.",
                                    style={
                                        "fontSize": "0.92rem",
                                        "color": C_MUTED,
                                        "lineHeight": "1.85",
                                        "marginBottom": "0",
                                    },
                                ),
                            ]
                        ),
                        className="border-0 shadow-sm",
                        style={
                            "borderRadius": "18px",
                            "background": "linear-gradient(180deg, #FFFFFF 0%, #FBFCFE 100%)",
                        },
                    ),
                    width=12,
                ),
                className="pb-5",
            ),
        ],
        fluid=True,
        className="px-4",
        style={"backgroundColor": C_BG, "minHeight": "100vh"},
    )


def _avatar(image_path=None, initials=""):
    avatar_style = {
        "width": "132px",
        "height": "132px",
        "borderRadius": "50%",
        "objectFit": "cover",
        "border": f"5px solid {C_SOFT}",
        "boxShadow": "0 8px 24px rgba(15,36,71,0.10)",
    }

    if image_path:
        return html.Img(src=image_path, style=avatar_style)

    return html.Div(
        initials,
        style={
            **avatar_style,
            "display": "inline-flex",
            "alignItems": "center",
            "justifyContent": "center",
            "backgroundColor": C_SOFT,
            "color": C_BLUE,
            "fontSize": "2rem",
            "fontWeight": "900",
        },
    )


def _team_card(
    name,
    role,
    description,
    contribution,
    tags,
    linkedin_url=None,
    accent_color=C_BLUE,
    image_path=None,
    initials="",
):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    _avatar(image_path=image_path, initials=initials),
                    style={"marginBottom": "20px"},
                ),
                html.H4(
                    name,
                    style={
                        "fontWeight": "900",
                        "color": C_NAVY,
                        "marginBottom": "6px",
                    },
                ),
                html.Div(
                    role,
                    style={
                        "fontSize": "0.92rem",
                        "fontWeight": "700",
                        "color": accent_color,
                        "marginBottom": "16px",
                    },
                ),
                html.P(
                    description,
                    style={
                        "fontSize": "0.90rem",
                        "color": C_MUTED,
                        "lineHeight": "1.8",
                        "marginBottom": "16px",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            "Aporte en el proyecto",
                            style={
                                "fontSize": "0.78rem",
                                "fontWeight": "800",
                                "letterSpacing": "0.05em",
                                "textTransform": "uppercase",
                                "color": "#7A8A9A",
                                "marginBottom": "8px",
                            },
                        ),
                        html.P(
                            contribution,
                            style={
                                "fontSize": "0.88rem",
                                "color": "#5D6D7E",
                                "lineHeight": "1.75",
                                "marginBottom": "18px",
                            },
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Span(
                            tag,
                            style={
                                "display": "inline-block",
                                "backgroundColor": C_SOFT,
                                "color": C_BLUE,
                                "padding": "6px 10px",
                                "borderRadius": "999px",
                                "fontSize": "0.78rem",
                                "fontWeight": "700",
                                "marginRight": "8px",
                                "marginBottom": "8px",
                            },
                        )
                        for tag in tags
                    ],
                    style={"marginBottom": "16px"},
                ),
                html.Div(
                    dbc.Button(
                        "Ver LinkedIn",
                        href=linkedin_url,
                        target="_blank",
                        color="primary",
                        outline=True,
                        style={
                            "borderRadius": "10px",
                            "fontWeight": "700",
                            "padding": "8px 16px",
                            "display": "inline-block" if linkedin_url else "none",
                        },
                    ),
                    style={"display": "block" if linkedin_url else "none"},
                ),
            ],
            style={"textAlign": "center", "padding": "32px 26px"},
        ),
        className="border-0 shadow-sm h-100",
        style={
            "borderRadius": "22px",
            "background": "linear-gradient(180deg, #FFFFFF 0%, #FBFCFE 100%)",
            "borderTop": f"4px solid {accent_color}",
        },
    )
