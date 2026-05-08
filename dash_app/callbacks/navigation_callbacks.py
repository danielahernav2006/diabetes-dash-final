from dash import Input, Output, State


def register_navigation_callbacks(app):

    @app.callback(
        Output("navbar-collapse", "is_open"),
        Input("navbar-toggler",   "n_clicks"),
        State("navbar-collapse",  "is_open"),
        prevent_initial_call=True,
    )
    def toggle_navbar(n, is_open):
        if n:
            return not is_open
        return is_open
