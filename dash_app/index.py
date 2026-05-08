import sys
import os

# ── sys.path setup (funciona en Windows y Linux) ──────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))           # dash_app/
_ROOT = os.path.dirname(_HERE)                               # diabetes_dash_project/
for _p in [_ROOT, _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

from dash import Input, Output, html, dcc
import dash_bootstrap_components as dbc

from dash_app.app import app
from dash_app.components.navbar import create_navbar
from dash_app.layouts.home import layout_home
from dash_app.layouts.dashboard import layout_dashboard
from dash_app.layouts.about import layout_about, register_about_callbacks
from dash_app.layouts.team import layout_team
from dash_app.layouts.model import layout_model
from dash_app.callbacks.dashboard_callbacks import register_dashboard_callbacks, DF_VIZ
from dash_app.callbacks.navigation_callbacks import register_navigation_callbacks
from dash_app.callbacks.model_callbacks import register_model_callbacks



# ── App shell ─────────────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    create_navbar(),
    html.Div(id="page-content"),
], style={"fontFamily": "Inter, sans-serif", "backgroundColor": "#f8f9fa", "minHeight": "100vh"})


# ── Page routing ──────────────────────────────────────────────────────────────
#@app.callback(Output("page-content", "children"), Input("url", "pathname"))
#def display_page(pathname):
  #  if pathname in ("/dashboard",):
 #       return layout_dashboard()
  #  if pathname in ("/about",):
  #      return layout_about()
  #  if pathname in ("/team",):
 #       return layout_team()
 #   if pathname in ("/predict", "/model", "/modelo"):
  #      return layout_model()
  #  return layout_home()


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname in ("/dashboard",):
        return layout_dashboard()
    if pathname in ("/about",):
        return layout_about()
    if pathname in ("/predict",):
        return layout_model()
    if pathname in ("/team",):
        return layout_team()
    return layout_home()


# ── Register callbacks ────────────────────────────────────────────────────────
register_dashboard_callbacks(app)
register_navigation_callbacks(app)
register_about_callbacks(app, DF_VIZ)
register_model_callbacks(app)


server = app.server

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
