import dash
from dash import  Input, Output, State, dcc
from dash.exceptions import PreventUpdate

import numpy as np
import pandas as pd

import os
import faiss

from datetime import datetime

from functions.dash_functions import table_type, clean_question_answer

def data_callbacks(app,tickets_assist,model,index):
    @app.callback(
        [Output('layout1_table', 'data'), Output('layout1_table', 'columns')],#output of the function
        [Input('layout1_load_button', 'n_clicks')],  # ID from the button
        [State('layout1_input_text', 'value')]  # Use this agregated value to work with it
    )
    def update_table(n_clicks, value):
        
        if n_clicks is None or value is None or value.strip() == "":
            raise PreventUpdate # don't run the query for nothing

        print(value)
        # Define a new data set order by similarity
        newdoc = clean_question_answer(value) #get the value and preprocessing

        new_doc_embedding = model.encode([newdoc], convert_to_tensor=True).cpu().numpy().astype(np.float32)
        faiss.normalize_L2(new_doc_embedding)  # normalisation, norme = 1

        K = 10000  # nb of neighbourgs
        D, I = index.search(new_doc_embedding, K)  # D: Distances (cosine similarity), I: index

        new_data = tickets_assist.iloc[I[0]].copy() 
        new_data['score'] = D[0].round(2)
        columns = [{'name': i, 'id': i, 'type': table_type(new_data[i])} for i in new_data.columns]

        return new_data.to_dict('records'), columns

    return dash.no_update

def export_csv_callbacks(app):
    @app.callback(
        Output('download-csv', 'data'),
        [Input('export-csv-button', 'n_clicks')],
        [State('layout1_table', 'derived_virtual_data'),
         State('layout1_table', 'derived_virtual_selected_rows')]
    )
    def generate_csv(n_clicks, all_rows_data, slctd_row_indices):#derived_virtual_data,derived_virtual_selected_rows
        if n_clicks is None:
            # Prevent downloading data until clicked
            raise PreventUpdate # don't run the query for nothing

        # Convert all displayed data in a dataframe
        dff = pd.DataFrame(all_rows_data)

        # Select only filtered rows, else take all data
        if slctd_row_indices is not None and len(slctd_row_indices) > 0:
            dff = dff.iloc[slctd_row_indices]

        y = datetime.now().year
        mo = datetime.now().month
        d = datetime.now().day
        h = datetime.now().hour
        m = datetime.now().minute
        s = datetime.now().second
        # Directly return the CSV as a download
        return dcc.send_data_frame(dff.to_csv, f"{y}-{mo}-{d}_{h}-{m}-{s}-filtred_data.csv", index=False)