import dash
from dash import dcc, html, Input, Output, dash_table, State
import numpy as np
import pandas as pd

import pathlib
import os,io,re
import joblib

from deep_translator import GoogleTranslator

from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

import faiss 

from tqdm import tqdm

from pages.main_layout import serve_layout
from callbacks.callbacks1 import data_callbacks,export_csv_callbacks
from functions.sbert_minibatch import *

if __name__ == '__main__':

    #Create an app like an class instance
    app = dash.Dash(__name__, meta_tags=[{"name": "Assist tickets", "content": "width=device-width, initial-scale=1"}],)

    app.title = "Dashboad Assist tickets"

    #Launch the server
    server = app.server

    #Load data
    DATA_PATH = os.path.join(os.path.dirname(os.getcwd()),"data/hotline_data/hotline_data_21_24.csv") 
    columns = ['request_n', 'start_date', 'en_question', 'answer', 'service', 'part_n', 'model', 'serial_n'] #en_question
    tickets_assist = pd.read_csv(DATA_PATH,encoding='utf-8')

    tickets_assist = tickets_assist[columns]
    
    tickets_assist['start_date'] = pd.to_datetime(tickets_assist['start_date']).dt.date.apply(lambda x: x.strftime('%Y-%m-%d')) # convert in a date format and after to a string 
    print('data loaded')

    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    # Load a pre-trained Sentence-BERT model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    # Put the model in the device
    model.to(device)
    print('Loaded Sentence-BERT model.')


    path_index_faiss = os.path.join(os.getcwd(),"model/sbert/faiss_index_hotline_21_24.index") #faiss_index_hotline_2023.index
    # load the matrix with sentence embeddings
    index = faiss.read_index(path_index_faiss)
    print(f"FAISS Index load with {index.ntotal} array inside.")
    
    ###layouts
    app.layout = serve_layout(tickets_assist)

    ###callbacks
    data_callbacks(app,tickets_assist,model,index) 
    export_csv_callbacks(app)

    #run application
    app.run_server(debug=True)


