import os,re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import time

from transformers import BertTokenizer, BertModel
import torch

from deep_translator import GoogleTranslator

from concurrent.futures import ThreadPoolExecutor, as_completed

## Function 

def translate(text_from, target_lang='en'):
    """"function use to translate a string text in english"""
    if pd.isna(text_from):
        return " "
    elif not isinstance(text_from, str):
        try:
            text_from = str(text_from)
        except:
            return " "
    try:
        translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text_from)
        return translated_text
    except Exception as e:
        #print(e)
        print(text_from)
        return text_from

def save_translated_questions(result, export_path,file_name):
    """Convert a list into a dataframe and save it """
    df = pd.DataFrame(result, columns=['TranslatedQuestion'])
    # export path 
    df.to_csv(os.path.join(export_path,file_name), index=False)


if __name__ == "__main__":
    t0 = time.time()
    # data loading
    DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), "data/chunks")
    EXPORT_PATH = os.path.join(os.path.dirname(os.getcwd()), "data/translated_data")

    try:
        data_text = pd.read_csv(os.path.join(DATA_PATH,"chunk_2023.csv"))
        print("Data loaded")
    except Exception as e:
        print(e)

    #get all questions in a list
    #questions = data_text.question.values.tolist()
    answers = data_text.answer.values.tolist()
    #print(len(questions))
    print(len(answers))
    #questions = questions[:50]
    answers = answers[:50]
    results = []

    """for i in tqdm(range(len(questions))):
        results.append(translate(questions[i]))"""
    
    for i in tqdm(range(len(answers))):
        results.append(translate(questions[i]))

    # Après avoir obtenu tous les résultats, sauvegardez-les au chemin spécifié
    save_translated_questions(results, EXPORT_PATH,"questions_english_2023.csv")
    save_translated_questions(questions, EXPORT_PATH,"questions_2023.csv")
    print(time.time()-t0)