from flask import Flask, request,render_template
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask_pymongo import PyMongo
from pymongo import MongoClient

import pickle
nltk.download('vader_lexicon')

app = Flask(__name__)

# app.config['MONGO_URI'] = 'mongodb://localhost:27017/Mini_Project'
# client = MongoClient()
# db = client['Mini_Project']

# mongo = PyMongo(app)

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
        # "Sentiment": text['sentiment']

        # mongo.db.sakec.insert_many([{"Text": text['text']}])

    sentiment_score = sid.polarity_scores(text)

    data1 = {"Text": text, "Sentiment": sentiment}

    db.sakec.insert_one(data1)
    # data2 = {"Sentiment": sentiment}

    # data2 = {"$set": {"Sentiment": sentiment}}


    # db.sakec.insert_many({data1},{data2})

    

    # mongo.db.sakec.insert_one([{"Text": data['text']}])

    # sakec = sakec.insert_one({"Text": "text"})
    
    return render_template('analysis.html',sentiment=sentiment,text=text)

    # data2 = {"Sentiment": sentiment}
    # db.sakec.insert_one(data2)


if __name__ == "__main__":
    app.run(debug=True)


# from flask import Flask, request,jsonify,render_template
# import pickle
# from nltk.sentiment import SentimentIntensityAnalyzer

# app = Flask(__name__)

# # load the trained model
# with open('svm_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# # define a function for sentiment analysis
# def predict_sentiment(text):
#     """
#     Predict the sentiment of the input text using both a trained model and the SentimentIntensityAnalyzer.

#     Args:
#         text (str): The text to analyze.

#     Returns:
#         str: The predicted sentiment ('positive', 'negative', or 'neutral').
#     """
#     # use the trained model to predict sentiment
#     svm_model = model.predict([text])[0]

#     # use the SentimentIntensityAnalyzer to predict sentiment
#     sia = SentimentIntensityAnalyzer()
#     sia_scores = sia.polarity_scores(text)
#     sia_compound = sia_scores['compound']

#     # combine the model and SIA predictions to get the final sentiment
#     if model_prediction == 1 and sia_compound >= 0.05:
#         sentiment = 'positive'
#     elif model_prediction == 0 and sia_compound <= -0.05:
#         sentiment = 'negative'
#     else:
#         sentiment = 'neutral'

#     return sentiment

# # define the API endpoint for sentiment analysis
# @app.route('/predict_sentiment', methods=['POST'])
# def predict():
#     """
#     Endpoint for sentiment analysis.

#     Expects a JSON payload of the form:
#     {
#         "text": "The movie was great!"
#     }

#     Returns a JSON response of the form:
#     {
#         "sentiment": "positive"
#     }
#     """
#     data = request.get_json()
#     text = data['text']
#     sentiment = predict_sentiment(text)
#     return {'sentiment': sentiment}

# if __name__ == '_main_':
#     app.run(debug=True)