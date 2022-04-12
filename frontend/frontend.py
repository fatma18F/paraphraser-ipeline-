# Authors: David Francisco <Dfrancisco1998@gmail.com>
# Copyright (C) 2021 David Francisco and DynaGroup i.T. GmbH

"""
This script contains frontend.
"""

import plotly.express as px
from jupyter_dash import JupyterDash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
from dash import html
from dash import dcc

# Iris bar figure
def drawFigure():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=px.bar(
                        df, x="sepal_width", y="sepal_length", color="species"
                    ).update_layout(
                        template='plotly_dark',
                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    ),
                    config={
                        'displayModeBar': False
                    }
                )
            ])
        ),
    ])

# Text field
def drawText():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H2("Text"),
                ], style={'textAlign': 'center'})
            ])
        ),
    ])

# Data
df = px.data.iris()

# Build App
app = JupyterDash(external_stylesheets=[dbc.themes.SLATE])

app.layout = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dcc.Input(id='input-1-submit', type='text', placeholder='Enter a sentence to be paraphrased'),
                html.Button('Paraphrase This Sentence', id='btn-submit'),
                html.Label('Output'), html.Br(),html.Br(),
                html.Div(id='output-submit'),
            ], align='center'),
            html.Br(),
            html.Br(),
        ]), color = 'dark'
    )
])

@app.callback(Output('output-submit', 'children'),
                [Input('btn-submit', 'n_clicks')],
                [State('input-1-submit', 'value')])
def update_output(clicked, input1):
    if clicked:
        num_beams = 10
        num_return_sequences = 1
        return "T5 Paraphrase coming soon"

app.run_server(mode='external')
