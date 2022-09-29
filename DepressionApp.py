%%writefile DepressionApp.py

######################
# Import libraries
######################
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
import nltk
#import time
#import seaborn as sb
#import matplotlib.pyplot as plt

######################
# Custom function
######################
## Working on the text processing function

data1 = pd.read_csv('clean_d_tweets.csv')
data2 = pd.read_csv('clean_non_d_tweets.csv')
data1.drop(['id', 'conversation_id', 'created_at', 'date', 'timezone', 'place', 'hashtags', 'cashtags', 'user_id', 'user_id_str'
           , 'name', 'day', 'hour', 'link', 'urls', 'photos', 'video',
       'thumbnail', 'retweet', 'nlikes', 'nreplies', 'nretweets', 'quote_url',
       'search', 'near', 'geo', 'source', 'user_rt_id', 'user_rt',
       'retweet_id', 'reply_to', 'retweet_date', 'translate', 'trans_src',
       'trans_dest' ], 1, inplace= True)
#Label = pd.DataFrame(np.random.choice(np.array(['Depressed']),3082).transpose())
#Label = Label.transpose()
#Label = pd.DataFrame(Label)
#Label = Label.rename(columns= {0: 'Label'})
#data1 = data1.join(Label)
data2.drop(['id', 'conversation_id', 'created_at', 'date', 'timezone', 'place', 'hashtags', 'cashtags', 'user_id', 'user_id_str'
           , 'name', 'day', 'hour', 'link', 'urls', 'photos', 'video',
       'thumbnail', 'retweet', 'nlikes', 'nreplies', 'nretweets', 'quote_url',
       'search', 'near', 'geo', 'source', 'user_rt_id', 'user_rt',
       'retweet_id', 'reply_to', 'retweet_date', 'translate', 'trans_src',
       'trans_dest', 'username' ], 1, inplace= True)
Label2, Label = pd.DataFrame(np.random.choice(np.array(['Not_Depressed']), 4687).transpose()), pd.DataFrame(np.random.choice(np.array(['Depressed']),3082).transpose())
Label, Label2 = Label.rename(columns= {0: 'Label'}), Label2.rename(columns= {0: 'Label'})

#Label2 = Label2.transpose()
#Label2 = pd.DataFrame(Label2)
#Label2 = Label2.rename(columns= {0: 'Label'})
data2, data1 = data2.join(Label2), data1.join(Label)
data1.dropna(inplace= True)
data2.dropna(inplace= True)
data1.drop(['language', 'username'], 1, inplace= True)
data2.drop('language', 1, inplace= True)
combined_data = pd.concat([data1, data2], 0, ignore_index= True )

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
# Input molecules (Side Panel)
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
#X = generate(SMILES)
#X[1:] # Skips the dummy first item

######################
# Pre-built model
######################

# Reads in saved model
load_model = pickle.load(open('model.pkl', 'rb'))
# Apply model to make predictions
prediction = load_model.predict(tfidff)
prediction


#st.header('Predicted Tweets')
#prediction[1:] # Skips the dummy first item
