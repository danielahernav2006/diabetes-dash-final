from dash import html, dash_table
import dash_bootstrap_components as dbc

C_BLUE = "#1B3A6B"
C_RED  = "#C0392B"


def create_summary_table_section():
    return dbc.Card([
        dbc.CardHeader(
            html.Div([
                html.P("Detalle por categoría",
                       style={"fontSize": "0.75rem", "fontWeight": "600",
                              "textTransform": "uppercase", "letterSpacing": "0.06em",
                              "color": "#7F8C8D", "marginBottom": "2px"}),
                html.H6("Distribución del factor seleccionado según estado de diabetes",
                        className="mb-0",
                        style={"fontWeight": "600", "color": "#2C3E50"}),
            ]),
            style={"backgroundColor": "#FAFBFC", "border": "none",
                   "paddingTop": "14px", "paddingBottom": "14px"},
        ),
        dbc.CardBody([
            dash_table.DataTable(
                id="summary-table",
                columns=[],
                data=[],
                style_table={"overflowX": "auto"},
                style_cell={
                    "fontFamily": "Inter, Arial, sans-serif",
                    "fontSize": "12.5px",
                    "padding": "9px 14px",
                    "textAlign": "center",
                    "border": "1px solid #ECEFF1",
                    "color": "#2C3E50",
                },
                style_header={
                    "backgroundColor": C_BLUE,
                    "color": "white",
                    "fontWeight": "600",
                    "textAlign": "center",
                    "fontSize": "12px",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.04em",
                    "border": "none",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"},
                     "backgroundColor": "#F7F9FC"},
                    {"if": {"filter_query": '{Categoría} contains "Diabetes"'},
                     "backgroundColor": "#FEF0EE", "fontWeight": "600"},
                ],
                page_size=12,
                sort_action="native",
            )
        ], className="p-0"),
    ], className="border-0 shadow-sm mb-4",
       style={"borderRadius": "10px", "overflow": "hidden"})
