import logging
import os,re
import datetime

import pandas as pd
import numpy as np
import joblib,pathlib

from deep_translator import GoogleTranslator

from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.pipeline import make_pipeline


from sklearn.metrics.pairwise import cosine_similarity

## Fonctions

def find_similar(tfidf_matrix, vector):
  """return top_n sentences clossiest to the main sentence thanks to the cosine_similarity of tow arrays"""
  cosine_similarities = cosine_similarity(vector, tfidf_matrix).flatten()

  docs_index = cosine_similarities.argsort()[::-1] #liste of index ordered by the cosine value in decreas order

  return docs_index, cosine_similarities[docs_index]


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
  text_from = re.sub('[%s]' % re.escape(""""#$£%&*§+,-/<=>@[\]^_`{|}~"""), ' ', text_from)  # remove punctuations
  text_from = re.sub(r'\xa0', ' ', text_from) # delete insecable space
  text_from = re.sub('\s+', ' ', text_from).strip()  # remove extra whitespace
  text_from = text_from.lower() #change to lowercase

  return text_from

def homemade_tokenizer(text):
  # Utiliser une expression régulière pour matcher les mots et numéros de série
  # \w+ match word
  # \d+ match number
  token_pattern = r"(?u)\b\w+(?:[-']\w+)*\b|\b\d+(?:[-]\d+)*\b"
  return re.findall(token_pattern, text)

##Main
if __name__ == "__main__":

  #temporal datas
  y = datetime.datetime.now().year
  mo = datetime.datetime.now().month
  d = datetime.datetime.now().day
  h = datetime.datetime.now().hour
  m = datetime.datetime.now().minute
  s = datetime.datetime.now().second

  path_log = os.getcwd() + "/log_tf_idf"
  #creat a log file in the log folder
  os.makedirs(path_log, exist_ok=True)
  logging.basicConfig(filename=os.path.join(path_log,f'execution_{y}-{mo}-{d}_{h}-{m}-{s}'),level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')

  DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"data/chunks") # pathlib.Path("/workspaces/Assist_tickets/data/chunks")

  try:
    data_text = pd.read_csv(os.path.join(DATA_PATH,os.listdir(DATA_PATH)[0]),index_col=0,encoding='utf-8')
    for path_chunk in os.listdir(DATA_PATH)[1:]:
        data_text = pd.concat( [data_text,pd.read_csv(os.path.join(DATA_PATH,path_chunk),index_col=0,encoding='utf-8')],ignore_index=True)
    print('data loaded')
    logging.info("Data loaded and concatenated successfully.")

  except Exception as e:
    logging.error(f"Failed to load and concatenate data: {e}")
  
  #Train part 
      
  if not os.path.exists(os.path.join(os.path.dirname(os.getcwd()),"model/tf_idf/tfidf_matrix.joblib")):
    print('training part')
    tfidf_pipeline = tfidf_pipeline = make_pipeline(CountVectorizer(ngram_range=(1,2), strip_accents='unicode',
                                                                    analyzer='word',lowercase=True,
                                                                    tokenizer=homemade_tokenizer),
                                                                    TfidfTransformer())
    tfidf_pipeline.fit(data_text["question"])

    tfidf_matrix = tfidf_pipeline.transform(data_text["question"])

    
    joblib.dump(tfidf_pipeline, os.path.join(os.path.dirname(os.getcwd()),"model/tf_idf/tfidf_pipeline.joblib"))
    joblib.dump(tfidf_matrix, os.path.join(os.path.dirname(os.getcwd()),"model/tf_idf/tfidf_matrix.joblib"))

  else:
    print('training part already done')
    tfidf_pipeline = joblib.load( os.path.join(os.path.dirname(os.getcwd()),"model/tf_idf/tfidf_pipeline.joblib"))
    tfidf_matrix = joblib.load(os.path.join(os.path.dirname(os.getcwd()),"model/tf_idf/tfidf_matrix.joblib")) 
  
  ## Test part
  question = data_text.iloc[np.random.randint(len(data_text))].question

  #question= "As you are sitting on the truck from left to right, hose lengths are 3400 3350 3250 3260 3340 3500"

  test_set = [question] # we take a random example from the test set
  print(test_set,'\n')

  test_vector = tfidf_pipeline.transform(test_set)

  print(test_vector.shape)

  ind, score  = find_similar(tfidf_matrix, test_vector.reshape(1, -1))

  print(data_text.iloc[ind].head(5).question)

  print(score[0:5])


###Documetation
#https://en.wikipedia.org/wiki/Tf%E2%80%93idf
#https://datascientest.com/tf-idf-intelligence-artificielle
#https://www.learndatasci.com/glossary/tf-idf-term-frequency-inverse-document-frequency/
