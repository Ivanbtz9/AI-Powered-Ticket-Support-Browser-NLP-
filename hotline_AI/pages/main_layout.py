import dash
from dash import dcc, html, dash_table
import numpy as np
import pandas as pd


from functions.dash_functions import table_type

def serve_layout(tickets_assist): 

    layout1 = html.Div(
        children=[   
            html.Div(id='layout1_flexbox_title',
                children=[
                    html.Div(id='layout1_Conteneur_title',
                            children=[
                            html.H1(
                                children=["HOTLINE ASSIST TICKETS MANAGER WITH AI"],
                                style={
                                    'color': '#1434A4',  # Egyptian blue
                                    'fontSize': 35
                                }
                            ),
                            html.P(
                                children=["Analyze the similarity between questions from the Assist application especially for the hotline service."],
                                style={
                                    'color': '#000000',  # Black
                                    'fontSize': 18  # font 
                                }
                            ),
                        ],
                        style={'flex': 1}  # take space but not to much, keep space for the picture
                    ),
                    # Conteneur for the picture
                    html.Div(
                        children=[
                            html.Img(src='assets/logo.png', style={'height': '125px', 'width': 'auto'}) # load a picture from the asset folder
                        ],
                        style={'flex': 'none'}  # Do not extend the image more than its necessary size
                    )
                ],
                style={
                'margin': 20,
                'padding': '20px',
                'fontFamily': 'Arial, sans-serif',
                'borderRadius': '15px',
                'boxShadow': '0px 0px 10px #aaaaaa',
                'display': 'flex', 
                'alignItems': 'center',
                'borderRadius': '8px',
                'justifyContent': 'space-between'
                }
            ),
            html.Div([
                dcc.Textarea( 
                    id='layout1_input_text',
                    placeholder='Enter your question...',
                    style={
                        'width': '90%', 
                        'height': '100px',
                        'padding': '0px 5px',
                        'margin': '0px 30px',  
                        'border': '1px solid #ccc',
                        'borderRadius': '8px',
                        'box-sizing': 'border-box'  
                    }
                ),
                html.Button('Load', id='layout1_load_button', n_clicks=0, # create a button to run the query
                    style={
                        'backgroundColor': '#1434A4', # Egyptian blue
                        'color': 'white',
                        'height': '50px',
                        'padding': '8px 50px',
                        'border': 'none',
                        'borderRadius': '5px',
                        'cursor': 'pointer',
                        'outline': 'none',
                        'fontSize': '16px'}
                )
                ],
            style={
            'margin': 20,
            'padding': '20px',
            'borderRadius': '15px',
            'display': 'flex', 
            'alignItems': 'center',
            'borderRadius': '8px',
            'justifyContent': 'space-between'
            }),
            html.Div(children=[
                dash_table.DataTable(
                id='layout1_table',
                columns=[{'name': i, 'id': i, 'type': table_type(tickets_assist[i])} for i in tickets_assist.columns], # get all columns
                data=tickets_assist.to_dict('records'),
                page_size=4,
                style_table={'overflowX': 'auto', 'textAlign': 'left'},
                style_cell= {'backgroundColor': 'white',
                            'color': 'black',
                            'fontSize': '15px',
                            'padding': '5px',  # ajust for more text 
                            'whiteSpace': 'normal',  # Allow text to wrap
                            'height': 'auto',  # ajuste the height of cell
                            #'minWidth': '80px',  # min width for each cell
                            'width': 'auto',  # width for each cell
                            #'maxWidth': '200px',  # max width for each cell
                            'textAlign': 'left'  # text alignment
                },
                style_data_conditional=[ # have different color for rows
                    {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'},
                    {'if': {'row_index': 'even'}, 'backgroundColor': 'rgb(220, 220, 220)'}
                ],
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_cell_conditional=[
                {'if': {'column_id': 'question'},'minWidth': '300px'},  # specific width for the column 'question'
                {'if': {'column_id': 'request_n'},'maxWidth': '200px'},
                {'if': {'column_id': 'start_date'},'maxWidth': '300px'},
                {'if': {'column_id': 'answer'},'minWidth': '300px'},], # specific width for the column 'answer'
                filter_action='native',
                sort_action='native',
                sort_mode='multi'
                ),
                html.Div(style={'marginTop': '90px'})], # make a space between the table and the other part 
                style={
                    'margin': 20,
                    'padding': '20px',
                    'justifyContent': 'space-between'
                    }),
            html.Div([html.Button('export CSV', id='export-csv-button',style={
                        'color': 'black',
                        'height': '50px',
                        'weight' : '80px',
                        'padding': '10px 10px',
                        'border': 'none',
                        'cursor': 'pointer',
                        'outline': 'none',
                        'fontSize': '16px'}),
                    dcc.Download(id='download-csv'),
                ])           
        ],
        style={
                'alignItems': 'center',
                'justifyContent': 'space-between',
                }
    )
    return layout1