from flask import Flask, request,render_template
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask_pymongo import PyMongo
from pymongo import MongoClient

import pickle
nltk.download('vader_lexicon')

app = Flask(__name__)

with open('svm_model.pkl', 'rb') as file:  
    model = pickle.load(file)


app.config['MONGO_URI'] = 'mongodb://localhost:27017/Mini_Project'
client = MongoClient()
db = client['Mini_Project']
collection = db["sakec"]

mongo = PyMongo(app)

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/analysis")
def analysis():
    return render_template('analysis.html')

@app.route('/analysis',methods=['GET','POST'])
def sid():
    # text = request.form['text']
    text = request.form['text']

    # mongo.db.sakec.insert_one([{"Text": text['text']}])


    sid = SentimentIntensityAnalyzer()
    sentiment_score = sid.polarity_scores(str(text))
    sentiment = sentiment_score['compound']
    if sentiment >= 0.05:
        sentiment = 'positive'
        print("Positive")
    elif sentiment <= -0.05:
        sentiment = 'negative'
        print("Negative")
    else:
        sentiment = 'neutral'
        print("Neutral")

    sentiment_score = sid.polarity_scores(text)

    data1 = {"Text": text, "Sentiment": sentiment}

    db.sakec.insert_one(data1)
    
    return render_template('analysis.html',sentiment=sentiment,text=text)

    # data2 = {"Sentiment": sentiment}
    # db.sakec.insert_one(data2)


if __name__ == "__main__":
    app.run(debug=True)
