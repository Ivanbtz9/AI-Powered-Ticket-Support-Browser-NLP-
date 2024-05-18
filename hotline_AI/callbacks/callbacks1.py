import dash
from dash import  Input, Output, State, dcc
from dash.exceptions import PreventUpdate
from dash import callback_context

import numpy as np
import pandas as pd

import os
import faiss

from datetime import datetime

from functions.dash_functions import table_type, clean_question_answer


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
    
def reset_filters_table(app):
    @app.callback(
        Output('layout1_table', 'filter_query'),
        Input('reset-filters-button', 'n_clicks')
    )
    def reset_filters(n_clicks):
        if n_clicks > 0:
            return ""  # return an empty list to reset filters
        return dash.no_update  # do nothing if you don't click on the button
            



def table_callbacks(app, tickets_assist, model, index):
    @app.callback(
        [Output('layout1_table', 'data'), Output('layout1_table', 'columns')],
        [Input('layout1_load_button', 'n_clicks'),
         Input('reset-data-button', 'n_clicks')],
        [State('layout1_input_text', 'value')]
    )
    def update_or_reset_table(load_clicks, reset_clicks, value):
        ctx = callback_context 
        #callback_context is a special object in Dash that provides context about the inputs that triggered the callback. 
        #This context is essential for understanding which component and what action (ex : button click) initiated the callback when you have multiple inputs and states involved.
        
        if not ctx.triggered:
            raise PreventUpdate

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'layout1_load_button' and value:
            print(value)
            # update the table here
            newdoc = clean_question_answer(value)
            new_doc_embedding = model.encode([newdoc], convert_to_tensor=True).cpu().numpy().astype(np.float32)
            faiss.normalize_L2(new_doc_embedding)
            K = 30000 #taket the 30k nearest neighboors
            D, I = index.search(new_doc_embedding, K)
            new_data = tickets_assist.iloc[I[0]].copy()
            new_data['score'] = np.round(D[0], decimals=2)

            for col in new_data.select_dtypes(include=['float']).columns:
                new_data[col] = new_data[col].map(lambda x: f"{x:.2f}")

        elif button_id == 'reset-data-button':
            # reset the table here
            new_data = tickets_assist.copy()
        else:
            raise PreventUpdate

        columns = [{'name': i, 'id': i, 'type': table_type(new_data[i])} for i in new_data.columns]
        return new_data.to_dict('records'), columns
    

#################################### OLD CODE ##############################################

#
#def reset_table(app,tickets_assist):
#    @app.callback(
#        [Output('layout1_table', 'data'), Output('layout1_table', 'columns')],#output of the function
#        [Input('reset-data-button', 'n_clicks')],  # ID from the button
#    )
#    def update_table(n_clicks, value):
#        
#        if n_clicks is None or value is None or value.strip() == "":
#            raise PreventUpdate # don't run the query for nothing
#
#        columns = [{'name': i, 'id': i, 'type': table_type(tickets_assist[i])} for i in tickets_assist.columns]
#        #print(new_data.to_dict('records'))
#        return tickets_assist.to_dict('records'), columns



#def data_callbacks(app,tickets_assist,model,index):
#    @app.callback(
#        [Output('layout1_table', 'data'), Output('layout1_table', 'columns')],#output of the function
#        [Input('layout1_load_button', 'n_clicks')],  # ID from the button
#        [State('layout1_input_text', 'value')]  # Use this agregated value to work with it
#    )
#    def update_table(n_clicks, value):
#        
#        if n_clicks is None or value is None or value.strip() == "":
#            raise PreventUpdate # don't run the query for nothing
#
#        print(value)
#        # Define a new data set order by similarity
#        newdoc = clean_question_answer(value) #get the value and preprocessing
#
#        new_doc_embedding = model.encode([newdoc], convert_to_tensor=True).cpu().numpy().astype(np.float32)
#        faiss.normalize_L2(new_doc_embedding)  # normalisation, norme = 1
#
#        K = 30000  # nb of neighbourgs
#        D, I = index.search(new_doc_embedding, K)  # D: Distances (cosine similarity), I: index
#
#        new_data = tickets_assist.iloc[I[0]].copy()
#        new_data['score'] = np.round(D[0], decimals=2)
#        new_data = new_data.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
#
#        columns = [{'name': i, 'id': i, 'type': table_type(new_data[i])} for i in new_data.columns]
#        #print(new_data.to_dict('records'))
#        return new_data.to_dict('records'), columns
#
#    return dash.no_update
