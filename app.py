import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 





def clean_tweet(tweet): # for cleaning the sentence
    tweet = re.sub(r'http\S+', '', tweet) # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet) # rempve bitly links
    tweet = tweet.strip('[link]') # remove [links]
    my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]\\\\\^_`{|}~•@#'
    tweet = re.sub('(RT\s@[A-Za-z0-9-_]+[A-Za-z0-9-_]+)', '', tweet) # remove retweet
    tweet = re.sub('(@[A-Za-z0-9-_]+[A-Za-z0-9-_]+)', '', tweet)
    tweet = tweet.lower() # lower case
    tweet = re.sub('['+my_punctuation + ']+', ' ', tweet) # strip punctuation
    tweet = re.sub('([0-9]+)', '', tweet) # remove numbers
    tweet = re.sub('amp', '', tweet) # remove amp
    tweet = re.sub('\s+', ' ', tweet) #remove double spacing
    return tweet



def main():
    st.sidebar.title("About")
    st.sidebar.info("Blue Ocean Project")
    st.title("Disaster tweets classifier")
    choice = st.selectbox("Classification type: ",
                     ['Source', 'Type', 'Informativeness'])
    user_text = st.text_input("Enter the tweet", max_chars=280)
    #button = st.button("Predict")
    
    if st.button("Predict"): 
        # removing punctuation
        #user_text = re.sub('[%s]' % re.escape(string.punctuation), '', user_text)
        user_text = clean_tweet(user_text)
            
        # Customizing stop words list
        stop_words = stopwords.words('english')
        newStopWords = ['ur','u','nd'] # new stop word
        remove_stopword = ['not','no','nor',"don","aren","couldn","didn","hadn","hasn","haven","isn","mustn","mightn","needn","shouldn",
                             "wasn","wouldn","won"] # stop words that we don't want
        stop_words.extend(newStopWords) # add new stop word
        stop_words = [OldStopWords for OldStopWords in stop_words if OldStopWords not in remove_stopword] # remove some stop words
            
        # tokenizing
        tokens = nltk.word_tokenize(user_text)
        # removing stop words
        stopwords_removed = [token.lower() for token in tokens if not token.lower() in set(stop_words)]
        # taking root word
        lemmatizer = WordNetLemmatizer() 
        lemmatized_output = []
        for word in stopwords_removed:
            lemmatized_output.append(lemmatizer.lemmatize(word))
        lemmatized_output = ' '.join(lemmatized_output)
         
        if choice == 'Source':
            source_model = pickle.load(open('pickle/source.pkl', 'rb'))
            out_put1 = source_model[:-1].transform([lemmatized_output])
            result = source_model[-1].predict(out_put1)[0]
            st.success(result)
        if choice == 'Type':
           source_model = pickle.load(open('pickle/type.pkl', 'rb'))
           out_put1 = source_model[:-1].transform([lemmatized_output])
           result = source_model[-1].predict(out_put1)[0]
           st.success(result)
        if choice == 'Informativeness':
           source_model = pickle.load(open('pickle/Informativeness.pkl', 'rb'))
           out_put1 = source_model[:-1].transform([lemmatized_output])
           result = source_model[-1].predict(out_put1)[0]
           st.success(result)


if __name__ =='__main__':
    main()
    
