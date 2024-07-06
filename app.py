import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from langdetect import detect
import nltk
from googletrans import Translator

# Ensure necessary corpora are downloaded
nltk.download('punkt')

# Title of the app
st.title("Enhanced Sentiment Analysis App")
st.write("This app uses TextBlob to perform sentiment analysis on your text input and provides additional features like word cloud visualization, language detection, and translation.")

# Text input
user_input = st.text_area("Enter your text here:")

# File upload
uploaded_file = st.file_uploader("Or upload a text file", type=["txt"])

text = user_input

# Process uploaded file
if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    st.write(f"Uploaded file content:\n{text}")

if text:
    # Perform sentiment analysis
    blob = TextBlob(text)
    sentences = blob.sentences
    sentiments = [sentence.sentiment.polarity for sentence in sentences]

    # Display the results
    st.subheader("Sentiment Analysis Results")
    st.write(f"Overall Polarity: {blob.sentiment.polarity}")
    st.write(f"Overall Subjectivity: {blob.sentiment.subjectivity}")

    if blob.sentiment.polarity > 0:
        st.write("Overall Sentiment: Positive")
    elif blob.sentiment.polarity < 0:
        st.write("Overall Sentiment: Negative")
    else:
        st.write("Overall Sentiment: Neutral")

    # Sentiment Distribution
    st.subheader("Sentiment Distribution")
    sentiment_labels = ['Positive', 'Neutral', 'Negative']
    sentiment_counts = [sum(1 for sentiment in sentiments if sentiment > 0),
                        sum(1 for sentiment in sentiments if sentiment == 0),
                        sum(1 for sentiment in sentiments if sentiment < 0)]
    sentiment_df = pd.DataFrame({'Sentiment': sentiment_labels, 'Count': sentiment_counts})

    st.bar_chart(sentiment_df.set_index('Sentiment'))

    # Word Cloud
    st.subheader("Word Cloud")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # Language Detection and Translation
    st.subheader("Language Detection and Translation")
    language = detect(text)
    st.write(f"Detected Language: {language}")

    if language != 'en':
        translator = Translator()
        translated_text = translator.translate(text, dest='en').text
        st.write("Translated Text:")
        st.write(translated_text)
    else:
        st.write("The text is already in English.")

else:
    st.write("Please enter some text or upload a file to analyze.")
