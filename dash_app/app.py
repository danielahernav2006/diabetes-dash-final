import dash
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    use_pages=False,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
    ],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Diabetes EDA Dashboard · BRFSS 2015"
server = app.server
