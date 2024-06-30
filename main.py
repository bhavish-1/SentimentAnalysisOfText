import pandas as pd
import numpy as np
import os
import importlib
from matplotlib import pyplot as plt
import warnings
import joblib
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


plt.style.use('default')
warnings.filterwarnings('ignore')

def clean_stopwords(text):
    english_stopwords = set(stopwords.words('english'))
    return ' '.join(word for word in text.split() if word not in english_stopwords)

def cleaning_punctuations(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def cleaning_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)

def cleaning_url(text):
    return re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', ' ', text)

def cleaning_numbers(text):
    return re.sub(r'[0-9]+', '', text)

def lemmatizer_on_word(tokens):
    lm = nltk.WordNetLemmatizer()
    lemmatized_tokens = [lm.lemmatize(word) for word in tokens]
    return lemmatized_tokens

def predict_sentiment(model, lemmatized_sentiment):
    prediction = model.predict([lemmatized_sentiment])
    return prediction[0] 


def main(text):
    
    sentiment = text
    
    # Converting the text to lowercase
    sentiment = sentiment.lower()
    
    # Removing the URLs
    sentiment = cleaning_url(sentiment)

    # Removing the numeric numbers
    sentiment = cleaning_numbers(sentiment)

    # Removing the punctuations
    sentiment = cleaning_punctuations(sentiment)

    # Removing the consequtive characters
    sentiment = cleaning_repeating_char(sentiment)

    # Removing the stopwords
    sentiment = clean_stopwords(sentiment)

    # Word tokenization of tweet
    tokens = word_tokenize(sentiment)

    # Applying the lemmatizer on sentiments
    lemmatized_tokens = lemmatizer_on_word(tokens)

    # Converting the tokens back to the string
    lemmatized_sentiment = " ".join(lemmatized_tokens)

    # importing the model 
    model = joblib.load('model.pkl')

    # prediction sentiment
    prediction = predict_sentiment(model, lemmatized_sentiment)

    return prediction

