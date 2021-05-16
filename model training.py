import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
import warnings
import re
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import joblib


df=pd.read_csv('Twitter Sentiments.csv')

# removing usernames i.e pattern @user
df['tweet']=df['tweet'].apply(lambda x: re.sub("@\S+","",x))

# removing special characters,numbers and punctuations and replacing with space
df['tweet']=df['tweet'].apply(lambda x: re.sub("[^a-zA-Z#]"," ",x))

# removing stopwords
sw=stopwords.words('english')
df['tweet']=df['tweet'].apply(lambda x:" ".join([word for word in x.split() if word not in (sw)]))

# tokenization to get individual words for the lemmatization process
# using spit instead of nltk to keep hastags together with the words
tokenized_tweet=df['tweet'].apply(lambda x:x.split())

# taking the root form of the words using lemmatization
tokenized_tweet=tokenized_tweet.apply(lambda x:[WordNetLemmatizer().lemmatize(word,pos='v') for word in x])

# after lemmatization rejoining the tokenized words into a sentence
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i]=" ".join(tokenized_tweet[i])

# updating the tweet in the df
df['tweet']=tokenized_tweet

#splitting data into X and y
X=df['tweet']
y=df['label']

#train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=101,test_size=0.2)

#  converting to Vector
tfidf_vector=TfidfVectorizer()
X_train_vector=tfidf_vector.fit_transform(X_train)
X_test_vector=tfidf_vector.transform(X_test)


# Data balancing by oversampling using SMOTE
from imblearn.over_sampling import SMOTE
oversample = SMOTE(k_neighbors=1)
# transform the dataset
X_train_vector, y_train = oversample.fit_resample(X_train_vector, y_train)


# trainig and testing the model
model=LogisticRegression()
model.fit(X_train_vector,y_train)
model_pred=model.predict(X_test_vector)

#saving the model
model_filename='twitter_model.pkl'
vector_filename='twitter_vector.pkl'
joblib.dump(model,model_filename)
joblib.dump(tfidf_vector,vector_filename)