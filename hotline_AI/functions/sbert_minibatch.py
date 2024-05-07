import logging
import os,re,tqdm
import datetime

import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

from deep_translator import GoogleTranslator

import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

import faiss #stock embeddings

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
    path_data_en = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"data/translated_data/data_english_2023.csv")

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

    # Put the model in the device
    model.to(device)

    # Get all questions in a list
    questions = tickets_en.question.values.tolist()
    questions = questions[:100] # for instance: limit to first 100 for demonstration

    # Initialisation de la liste pour stocker les embeddings
    all_embeddings = []

    # process by batch to save RAM
    batch_size = 64  # depends of your memory

    for start_index in tqdm(range(0, len(questions), batch_size)):
        # select the batch
        questions_batch = questions[start_index:start_index + batch_size]
        
        # return embedding for embeddings for the batch
        embeddings_batch = model.encode(questions_batch, convert_to_tensor=True, show_progress_bar=False)
        
        # Stock embeddings
        all_embeddings.extend(embeddings_batch) # extend caus ethe result is like this [[1,4,...,8]]

    # Convert to a numpy object 
    all_embeddings = np.vstack(all_embeddings)
    print(f"shape of all embeddings: {all_embeddings.shape}")


    embeddings = np.array(all_embeddings).astype(np.float32)

    # create a faiss object for cosin similarity
    dim_col = embeddings.shape[1] 
    index = faiss.IndexFlatIP(dim_col)  # IndexFlatIP 

    # Normalisation is importante to reduce the calculation time
    faiss.normalize_L2(embeddings)

    # add embeddings to the index
    index.add(embeddings)

    print(f"{index.ntotal} array add to the FAISS index.")

    newdoc = "Hello,I need MSDS document for this oil.Can you send me?Regards,Radosław[Info Piece]Machine serial number--Part number947973Type and description of the part (or doc)oilParts"

    # Supposons que vous avez un nouveau document et que vous avez généré son embedding
    new_doc_embedding = model.encode([newdoc], convert_to_tensor=True).numpy().astype(np.float32)
    faiss.normalize_L2(new_doc_embedding)  # Normalisation pour la similarité cosinus

    # Recherche des K plus proches voisins
    K = 2  # Nombre de voisins les plus proches à trouver
    D, I = index.search(new_doc_embedding, K)  # D: Distances, I: Indices des voisins dans l'index

    print(f"Indices des documents les plus similaires: {I}")
    print(f"Scores de similarité (cosinus): {D}")


#pip install faiss-cpu  # Pour une installation sans GPU
# ou
#pip install faiss-gpu  # Pour une installation avec support GPU

#RAG : https://docs.mistral.ai/guides/basic-RAG/

