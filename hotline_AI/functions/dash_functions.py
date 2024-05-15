import pandas as pd 
import os,re,tqdm
import datetime
import pandas as pd
import numpy as np

from deep_translator import GoogleTranslator


def table_type(df_column):
    """Use this function to ruturn the dtype of a column from a DataFrame """

    if isinstance(df_column.dtype, pd.DatetimeTZDtype):
        return 'datetime',
    elif (isinstance(df_column.dtype, pd.StringDtype) or
            isinstance(df_column.dtype, pd.BooleanDtype) or
            isinstance(df_column.dtype, pd.CategoricalDtype) or
            isinstance(df_column.dtype, pd.PeriodDtype)):
        return 'text'
    elif (isinstance(df_column.dtype, pd.SparseDtype) or
            isinstance(df_column.dtype, pd.IntervalDtype) or
            isinstance(df_column.dtype, pd.Int8Dtype) or
            isinstance(df_column.dtype, pd.Int16Dtype) or
            isinstance(df_column.dtype, pd.Int32Dtype) or
            isinstance(df_column.dtype, pd.Int64Dtype) or
            isinstance(df_column.dtype, pd.Float32Dtype) or
            isinstance(df_column.dtype, pd.Float64Dtype)):
        return 'numeric'
    else:
        return 'any'
    

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
