from concurrent.futures import ThreadPoolExecutor, as_completed #librairie to make multi threading
from tqdm import tqdm
from deep_translator import GoogleTranslator #API google

import os
import numpy as np
import pandas as pd 


def translate_text(text_from, target_lang='en'):
    """Translate a string text into English."""
    if pd.isna(text_from) or not isinstance(text_from, str):
        return " "
    try:
        translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text_from)
        return translated_text
    except:
        try:
            translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text_from[:400])
            return translated_text
        
        except Exception as e:
            print(f"Translation error: {e}")
            return " "

def translate_questions(questions):
    """Translate a list of questions using multi threading."""
    max_workers = os.cpu_count() #return the nb of cores
    translated_questions = {} #created an empty dico to keep in mind the place of each translation

    with ThreadPoolExecutor(max_workers = max_workers) as executor:
        futures = {i: executor.submit(translate_text, question) for i, question in enumerate(questions)} #make a dictionnary with {index:translation_to_execute} and sumit jobs to the executor
        for future in tqdm(as_completed(futures.values()), total=len(questions), desc="Translating"): #a future  is a waiting task  
            indx = [key for key, val in futures.items() if val == future][0] # items return a list of [(key,value),...,(key,value))]
            translated_question = future.result() #catch the result 
            translated_questions[indx] = translated_question 
    # Reconstruct the list of translated questions in the original order
    translated_questions_list = [translated_questions[i] for i in range(len(questions))]
    return translated_questions_list

def main():
    DATA_PATH = "/home/ibotcazou/Bureau/MANITOU/Assist_tickets/data/hotline_data_2"
    EXPORT_PATH = "/home/ibotcazou/Bureau/MANITOU/Assist_tickets/data/hotline_data"
    
    for file in tqdm(os.listdir(DATA_PATH)):

        try:
            data = pd.read_csv(os.path.join(DATA_PATH,file))
            print(f"{file} loaded")
        except Exception as e:
            print(f"Error loading data: {e}")
            return

        bad_index = data["en_question"][data["en_question"]==' '].index

        data_without_translation = data.loc[bad_index]
        
        column_to_tanslate = 'question' # It will be question all the time, but you could also try with answer 

        questions = data_without_translation[f"{column_to_tanslate}"].values.tolist()
        
        translated_questions = translate_questions(questions)

        data.loc[bad_index,"en_question"]= translated_questions

        # Save translated questions
        
        data.to_csv(os.path.join(EXPORT_PATH,file),index=False)

if __name__ == "__main__":
  #t0 = time.time()
  main()
  #print(time.time()-t0)
