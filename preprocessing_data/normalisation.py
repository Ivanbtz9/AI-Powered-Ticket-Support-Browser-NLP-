import os,re,sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from deep_translator import GoogleTranslator
import datetime

#LOG initialisation
##temporal datas
y = datetime.datetime.now().year
mo = datetime.datetime.now().month
d = datetime.datetime.now().day
h = datetime.datetime.now().hour
m = datetime.datetime.now().minute
s = datetime.datetime.now().second

##create a log file in the log folder
path_log = os.path.join(os.getcwd(),"log_treatment_data")
os.makedirs(path_log, exist_ok=True)
logging.basicConfig(filename=os.path.join(path_log,f'execution_{y}-{mo}-{d}_{h}-{m}-{s}'),
                    level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')


##Function
def clean_question_answer(text_from):
    """Use this function to clean up your question and your answer. This is the standardization part of NLP tasks"""
    if pd.isna(text_from):
        return " "

    elif not isinstance(text_from,str):
        try:
            text_from = str(text_from)
        except:
            logging.error(f"Error with format {e}")
            return " "

    #remove all pontuation aspect
    text_from = re.sub('http\S+\s*', ' ', text_from)  # remove URLs
    text_from = re.sub('#\S+', '', text_from)  # remove hashtags
    text_from = re.sub('@\S+', ' ', text_from)  # remove mentions
    text_from = re.sub('[%s]' % re.escape(""""%#$£%&*§+,-<=>@[\]^_`{|}~"""), ' ', text_from)  # remove punctuations
    text_from = re.sub(r'\xa0', ' ', text_from) # delete insecable space
    text_from = re.sub('\s+', ' ', text_from).strip()  # remove extra whitespace
    text_from = text_from.lower() #change to lowercase

    return text_from


if __name__ == "__main__":

    #DATA PATH
    path_data = os.path.join(os.path.dirname(os.getcwd()),"data/raw_hotline_data/2024_hotline_until_30_04.xlsx") 
    export_path_data = os.path.join(os.path.dirname(os.getcwd()),"data/clean_tickets/hotline") #folder path to

    #LOAD DATA
    tickets = pd.read_excel(path_data)
    print('data loaded')

    keep_cols_en = ['Request Number', 'Creation Date','Service Description','Description', 'Description.1', 'Part number', 'Model','Serial N°']#'Department (Full)', 'Time to Solve (mm)',, 'Satisfaction'
    
    #keep_cols_fr = ['N° de demande', 'Emise le','Libellé du service', 'Description','Réponse', 'Ref. pièce', 'Modèle', 'N° de série']#'Entité (complet)','Délai de résolution (min)', , 'Satisfaction'
    
    new_col_names = ["request_n", 'start_date', 'service','question','answer', 'part_n', 'model', "serial_n"] #'location',"satisfaction" "time_mm",

    print(tickets.columns)
    print(tickets.shape)

    #sys.exit()

    #tickets = tickets[keep_cols_fr] #selecte good columns 
    tickets = tickets[keep_cols_en]

    tickets.rename(columns={k:v for k,v in zip(keep_cols_en,new_col_names)},inplace=True) #rename columns 

    
    tickets = tickets.dropna(subset=['question']) #delete ligne where we don't have a question

    columns_to_clean = ["answer", "question", "part_n", "model", "serial_n"]

    #dictionary to change fr to en 
    #dico_fr_en = {'Référence / Prix / Délai':'p n price availability', 'Informations pièces':'parts informations',
    #              'Identification référence':'parts number identification', 
    #              'Non conformité technique de la pièce':'part non conformity (technical)','Information retour pièces':'spare parts return information'}
    
    #tickets.service = tickets.service.map(dico_fr_en) # that depend of the language version of your data, a way to translate quickly the service in english 

    for col in columns_to_clean:
        tickets[col] = tickets[col].apply(clean_question_answer) #Apply the function on each value of each column
        print("Cleaned column:", col)

    print(tickets.service.unique())
    tickets.to_csv(os.path.join(export_path_data,"chunk_2024_until_30_04.csv"))
    print(tickets.shape)

   
