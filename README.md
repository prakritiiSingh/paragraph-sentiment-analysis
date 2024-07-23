# Sentiment Analysis of Paragraphs

This project provides multiple methods to analyze the sentiment of paragraphs using various NLP libraries and models in Python.

## Prerequisites

Ensure you have Python installed. You will also need to install the required packages:

```sh
pip install nltk
pip install textblob
pip install scikit-learn
pip install transformers
pip install torch
Usage
VADER Sentiment Analysis
This method uses the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool.

python
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Example paragraphs
paragraphs = [
    "I love this new product! It has exceeded my expectations in every way. The quality is superb, and I will definitely recommend it to my friends.",
    "This experience was terrible. The service was slow, and the staff was rude. I will not be coming back here again.",
    "I'm not sure how I feel about this. On one hand, it has some good features, but on the other hand, it also has some major drawbacks.",
    "Today was the best day ever! Everything went perfectly, and I couldn't be happier.",
    "I can't stand this anymore. The constant noise and disruptions are driving me crazy."
]

# Analyze sentiment for each paragraph
for paragraph in paragraphs:
    scores = sid.polarity_scores(paragraph)
    print(f"Paragraph: {paragraph}\nScores: {scores}\n")
TextBlob Sentiment Analysis
This method uses TextBlob for sentiment analysis.

python
from textblob import TextBlob

# Example paragraphs
paragraphs = [
    "I love this new product! It has exceeded my expectations in every way. The quality is superb, and I will definitely recommend it to my friends.",
    "This experience was terrible. The service was slow, and the staff was rude. I will not be coming back here again.",
    "I'm not sure how I feel about this. On one hand, it has some good features, but on the other hand, it also has some major drawbacks.",
    "Today was the best day ever! Everything went perfectly, and I couldn't be happier.",
    "I can't stand this anymore. The constant noise and disruptions are driving me crazy."
]

# Analyze sentiment for each paragraph
for paragraph in paragraphs:
    blob = TextBlob(paragraph)
    sentiment = blob.sentiment
    print(f"Paragraph: {paragraph}\nSentiment: {sentiment}\n")
Naive Bayes Sentiment Analysis
This method uses Naive Bayes for sentiment classification.

python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample data
paragraphs = [
    "I love this new product! It has exceeded my expectations in every way. The quality is superb, and I will definitely recommend it to my friends.",
    "This experience was terrible. The service was slow, and the staff was rude. I will not be coming back here again.",
    "I'm not sure how I feel about this. On one hand, it has some good features, but on the other hand, it also has some major drawbacks.",
    "Today was the best day ever! Everything went perfectly, and I couldn't be happier.",
    "I can't stand this anymore. The constant noise and disruptions are driving me crazy."
]
labels = [1, 0, 0, 1, 0]  # 1 for positive sentiment, 0 for negative sentiment

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(paragraphs, labels, test_size=0.25, random_state=42)

# Create a pipeline that combines the vectorizer and classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Predict sentiment for new paragraphs
new_paragraphs = [
    "This product has changed my life for the better. Highly recommend it!",
    "The service was awful, and the food was inedible."
]
predicted_labels = model.predict(new_paragraphs)
print(f"New Paragraphs: {new_paragraphs}\nPredicted Labels: {predicted_labels}")
Transformer-based Sentiment Analysis
This method uses a pre-trained BERT model for sentiment analysis.

python
from transformers import pipeline

# Specify the model name explicitly
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Load the sentiment analysis pipeline with the specified model
sentiment_pipeline = pipeline('sentiment-analysis', model=model_name)

# Example paragraphs
paragraphs = [
    "I love this new product! It has exceeded my expectations in every way. The quality is superb, and I will definitely recommend it to my friends.",
    "This experience was terrible. The service was slow, and the staff was rude. I will not be coming back here again.",
    "I'm not sure how I feel about this. On one hand, it has some good features, but on the other hand, it also has some major drawbacks.",
    "Today was the best day ever! Everything went perfectly, and I couldn't be happier.",
    "I can't stand this anymore. The constant noise and disruptions are driving me crazy."
]

# Analyze sentiment for each paragraph
for paragraph in paragraphs:
    result = sentiment_pipeline(paragraph)
    print(f"Paragraph: {paragraph}\nResult: {result}\n")
Multilingual BERT Sentiment Analysis
This method uses a multilingual BERT model for sentiment analysis.

python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Load the sentiment analysis pipeline with the specified model
sentiment_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Example paragraphs
paragraphs = [
    "I love this new product! It has exceeded my expectations in every way. The quality is superb, and I will definitely recommend it to my friends.",
    "This experience was terrible. The service was slow, and the staff was rude. I will not be coming back here again.",
    "I'm not sure how I feel about this. On one hand, it has some good features, but on the other hand, it also has some major drawbacks.",
    "Today was the best day ever! Everything went perfectly, and I couldn't be happier.",
    "I can't stand this anymore. The constant noise and disruptions are driving me crazy."
]

# Analyze sentiment for each paragraph
for paragraph in paragraphs:
    result = sentiment_pipeline(paragraph)
    print(f"Paragraph: {paragraph}\nResult: {result}\n")
License
This project is licensed under the MIT License.
Feel free to adjust any part of the README file to better fit your
