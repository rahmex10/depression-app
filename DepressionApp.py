
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
import nltk

combined_data = pd.read_csv('combined_data.csv')

import string
from nltk.corpus import stopwords
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#txtproc = pickle.load(open('text_process.pkl', 'rb'))
#combined_data = pickle.load(open('combined_data.pkl', 'rb')) 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer= text_process).fit(combined_data['tweet'])


######################
# Page Title
######################

image = Image.open('depression.jpg')

st.image(image, use_column_width=True)

st.write("""
# Depression Predictor App
This app predicts the likelihood of a quotes to being **Depressive or not**
""")


######################
#(Side Panel)
######################

st.sidebar.header('User Input Tweets / Quotes')

## Read SMILES input
tweet_input = "I am very sad"

tweets = st.sidebar.text_area("Tweet input", tweet_input)
tweets = "C\n" + tweets #Adds C as a dummy, first item
tweets = tweets.split('\n')

st.header('Input Tweet / Quote' )
tweets[1:] # Skips the dummy first item

## Calculate depressive descriptors
st.header('Predicted Tweets / Quotes')
#cv = CountVectorizer(analyzer= text_process).fit(combined_data['tweet'])
#bagofwords = cv.transform(combined_data['tweet'])
#cv = CountVectorizer(analyzer= text_process).fit(tweets[1:])
bagofwords = cv.transform(pd.Series(tweets[1:]))
from sklearn.feature_extraction.text import TfidfTransformer
tfidff = TfidfTransformer().fit_transform(bagofwords)

######################
# Pre-built model
######################

# Reads in saved model
load_model = pickle.load(open('model.pkl', 'rb'))
# Apply model to make predictions
prediction = load_model.predict(tfidff)
prediction
