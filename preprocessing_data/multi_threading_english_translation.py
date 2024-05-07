import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed #librairie to make multi threading
from tqdm import tqdm
from deep_translator import GoogleTranslator #API google
import time

#Local path 
DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), "data/clean_tickets/hotline")
EXPORT_PATH = os.path.join(os.path.dirname(os.getcwd()), "data/hotline_data")


def translate_text(text_from, target_lang='en'):
    """Translate a string text into English."""
    if pd.isna(text_from) or not isinstance(text_from, str):
        return " "
    try:
        translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text_from)
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


def save_translated_questions(df,translated_questions, export_path, file_name):
    """Save translated questions to a CSV file."""
    #df = pd.DataFrame({"TranslatedQuestion": translated_questions})
    df["en_question"] = translated_questions # append a new colums into the dataframe df 
    df.to_csv(os.path.join(export_path, file_name), index=False)
    print(f"Translated questions saved to {os.path.join(export_path, file_name)}")


def main():

    for file in tqdm(os.listdir(DATA_PATH)):

        try:
            data_text = pd.read_csv(os.path.join(DATA_PATH,file))
            print(f"{file} loaded")
        except Exception as e:
            print(f"Error loading data: {e}")
            return
        
        column_to_tanslate = 'question' # It will be question all the time, but you could also try with answer 

        questions = data_text[f"{column_to_tanslate}"].values.tolist()
        #questions = questions[:1000]

        
        translated_questions = translate_questions(questions)

        # Save translated questions
        save_translated_questions(data_text,translated_questions, EXPORT_PATH, f"{file[:-4]}_with_en_question.csv")

if __name__ == "__main__":
  #t0 = time.time()
  main()
  #print(time.time()-t0)
