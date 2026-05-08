from dash import html
import dash_bootstrap_components as dbc

C_NAVY = "#0F2447"
C_BLUE = "#4A7FC1"
C_RED = "#C0392B"
C_MUTED = "rgba(255,255,255,0.68)"
C_SOFT = "rgba(255,255,255,0.42)"
C_BORDER = "#1A2F56"


def create_navbar():
    nav_link_style = {
        "color": C_MUTED,
        "fontSize": "0.88rem",
        "fontWeight": "600",
        "padding": "0.5rem 1rem",
        "borderRadius": "8px",
        "transition": "all 0.2s ease",
    }

    brand_title_style = {
        "color": "white",
        "fontWeight": "800",
        "fontSize": "0.98rem",
        "letterSpacing": "0.01em",
        "lineHeight": "1.1",
        "display": "block",
    }

    brand_subtitle_style = {
        "color": C_SOFT,
        "fontSize": "0.78rem",
        "fontWeight": "500",
        "lineHeight": "1.1",
        "display": "block",
        "marginTop": "2px",
    }

    return dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    [
                                        html.Div(
                                            style={
                                                "width": "10px",
                                                "height": "10px",
                                                "borderRadius": "50%",
                                                "backgroundColor": C_BLUE,
                                                "display": "inline-block",
                                                "marginRight": "6px",
                                                "boxShadow": "0 0 0 3px rgba(74,127,193,0.12)",
                                            }
                                        ),
                                        html.Div(
                                            style={
                                                "width": "10px",
                                                "height": "10px",
                                                "borderRadius": "50%",
                                                "backgroundColor": C_RED,
                                                "display": "inline-block",
                                                "boxShadow": "0 0 0 3px rgba(192,57,43,0.12)",
                                            }
                                        ),
                                    ],
                                    style={
                                        "display": "inline-flex",
                                        "alignItems": "center",
                                        "marginRight": "12px",
                                    },
                                ),
                                width="auto",
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.Span("Diabetes Analytics", style=brand_title_style),
                                        html.Span("BRFSS 2015", style=brand_subtitle_style),
                                    ]
                                ),
                                width="auto",
                            ),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    href="/",
                    style={"textDecoration": "none"},
                ),

                dbc.NavbarToggler(id="navbar-toggler"),

                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.NavItem(
                                dbc.NavLink(
                                    "Inicio",
                                    href="/",
                                    active="exact",
                                    style=nav_link_style,
                                )
                            ),
                            dbc.NavItem(
                                dbc.NavLink(
                                    "Dashboard",
                                    href="/dashboard",
                                    active="exact",
                                    style=nav_link_style,
                                )
                            ),
                            dbc.NavItem(
                                dbc.NavLink(
                                    "Análisis",
                                    href="/about",
                                    active="exact",
                                    style=nav_link_style,
                                )
                            ),
                            dbc.NavItem(
                                dbc.NavLink(
                                    "Predicción",
                                    href="/predict",
                                    active="exact",
                                    style=nav_link_style,
                                )
                            ),
                            dbc.NavItem(
                                dbc.NavLink(
                                    "Equipo",
                                    href="/team",
                                    active="exact",
                                    style=nav_link_style,
                                )
                            ),
                        ],
                        navbar=True,
                        className="ms-auto align-items-center",
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ],
            fluid=True,
            className="px-3 px-md-4",
        ),
        color=C_NAVY,
        dark=True,
        sticky="top",
        style={
            "boxShadow": "0 2px 10px rgba(0,0,0,0.16)",
            "borderBottom": f"1px solid {C_BORDER}",
            "paddingTop": "0.7rem",
            "paddingBottom": "0.7rem",
        },
    )
