
import joblib
from flask import Flask, request, render_template

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


app=Flask(__name__)


# Loading the model and tfidf vectorizer ffrom the disk

vector= joblib.load('twitter_vector.pkl')
model=joblib.load('twitter_model.pkl')

def Identification(tweet):
    vectorized=vector.transform([tweet])
    my_pred=model.predict(vectorized)

    if my_pred==1:
        return (['Tweet is Rascist/Sexist'])
    else:
        return (['Tweet is not Rascist/Sexist'])




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Identify',methods=['POST'])
def Identify():
    tweet=request.form['tweet']
    result=Identification(tweet)

    return render_template('index.html',identify_text=result)

if __name__=="__main__":
    app.run(debug=True)


