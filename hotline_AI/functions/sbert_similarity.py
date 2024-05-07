import logging
import os,re,tqdm
import datetime

import pandas as pd
import numpy as np

from deep_translator import GoogleTranslator

from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from tqdm import tqdm

##Function
def translate_text(text_from, target_lang='en'):
  """Translate a string text into English."""
  if pd.isna(text_from) or not isinstance(text_from, str):
      return ""
  try:
      translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text_from)
      return translated_text
  except Exception as e:
      print(f"Translation error: {e}")
      return text_from

def clean_question_answer(text_from):
  """Use this function to clean up your question and your answer. This is the standardization part of NLP tasks"""
  if pd.isna(text_from):
      return " "

  elif not isinstance(text_from,str):
      try:
          text_from = str(text_from)
      except Exception as e:
          logging.error(f"Error with format {e}")
          return " "

  #remove all pontuation aspect
  text_from = re.sub('http\S+\s*', ' ', text_from)  # remove URLs
  text_from = re.sub('#\S+', '', text_from)  # remove hashtags
  text_from = re.sub('@\S+', ' ', text_from)  # remove mentions
  text_from = re.sub('[%s]' % re.escape(""""#$£%&*§+,-/<=>@[\]^_`{|}~"""), ' ', text_from)  # remove punctuations
  text_from = re.sub(r'\xa0', ' ', text_from) # delete insecable space
  text_from = re.sub('\s+', ' ', text_from).strip()  # remove extra whitespace

  text_from = translate_text(text_from)
  text_from = text_from.lower()

  return text_from

if __name__ == '__main__':

    ##Load data in english
    path_data_en = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"data/translated_data/questions_english_2023.csv")

    # chose your device
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    #LOAD DATA
    tickets_en = pd.read_csv(path_data_en,index_col=0)
    print('data loaded')



    # Load a pre-trained Sentence-BERT model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    print('Loaded Sentence-BERT model.')

    # Get all questions in a list
    questions = tickets_en.TranslatedQuestion.values.tolist()
    questions = questions[:10]  # Example: limit to first 10 for demonstration

    # Generate embeddings for all questions
    # The model automatically handles batching internally, so no need for a DataLoader here
    embeddings = model.encode(questions, convert_to_tensor=True, show_progress_bar=True)

    print(f"Shape of all embeddings: {embeddings.shape}")

