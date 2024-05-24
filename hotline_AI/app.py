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

from pages.main_layout import *
from callbacks.callbacks1 import *
from functions.sbert_minibatch import *

if __name__ == '__main__':

    #Create an app like an class instance
    app = dash.Dash(__name__, meta_tags=[{"name": "Assist tickets", "content": "width=device-width, initial-scale=1"}],)

    app.title = "Dashboad Assist tickets"

    #Launch the server
    server = app.server

    #Load data
    csv_name = "data_21_24_hotline_modif23-05-24.csv"
    DATA_PATH = os.path.join(os.path.dirname(os.getcwd()),f"data/hotline_data/{csv_name}") 
    columns_to_keep = ['request_n', 'start_date', 'service', 'en_question', 'en_answer', 'part_n', 'model_modif2', 'serial_n'] #en_question
    new_columns_name = ['Ticket nb', 'Start date', 'Service', 'Question', 'Answer', 'Part nb', 'Model', 'Serial nb']
    tickets_assist = pd.read_csv(DATA_PATH,encoding='utf-8')

    tickets_assist = tickets_assist[columns_to_keep]
    tickets_assist.rename(columns={k:v for k,v in zip(columns_to_keep,new_columns_name)},inplace=True)
    
    tickets_assist['Start date'] = pd.to_datetime(tickets_assist['Start date']).dt.date.apply(lambda x: x.strftime('%Y-%m-%d')) # convert in a date format and after to a string 
    print('data loaded')

    #Launch the device to a GPU if possible
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    # Load a pre-trained Sentence-BERT model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    model.to(device)# Put the model in the device
    print('Loaded Sentence-BERT model.')

    # load the FAISS INDEX with sentence embeddings
    index_name = "faiss_index_hotline_21_24.index"
    path_index_faiss = os.path.join(os.getcwd(),f"model/sbert/{index_name}") #faiss_index_hotline_2023.index
    index = faiss.read_index(path_index_faiss)
    print(f"FAISS Index load with {index.ntotal} array inside.")
    
    ###layouts
    app.layout = serve_layout(tickets_assist)

    ###callbacks
    export_csv_callbacks(app)
    reset_filters_table(app)
    table_callbacks(app, tickets_assist, model, index)

    #run application
    app.run_server(debug=True)


"""
Exemples pour la présentation : 


1)J'ai une commande urgente avec une MRT, j'aurai besoin des passages de sangles de la radiocommande. Ces référence ne sont pas détaillés sur la vue éclatée, ainsi que les références des deux sangles possibles

MRT 2550

réponse :

52704862

2) Wayne asked for the blower resistor for the ATC climate control system

Blower resister
MLT 737

réponse :

958814

3) an you please provide the part number for the tie rod ends for this machine? 
M 50-4 S2

réponse :

894466

4) how can i find the correct part number for the alarm module


###########################

with a serial nb, ex : 759917

with a model name ex : sslm

with a part number ex : 730747

"""