# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from nltk import download
import re

# Download necessary NLTK resources if not already available
try:
    download('punkt')
    download('stopwords')
    download('wordnet')
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")

# Initialize lemmatizer and VADER analyzer
lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

# Function to clean and preprocess text
def clean_text(text):
    try:
        if not isinstance(text, str):
            text = str(text)

        # Remove HTML tags and URLs
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)

        # Normalize text
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)

        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        # Perform lemmatization
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        return ' '.join(tokens)
    except Exception as e:
        st.error(f"Error cleaning text: {e}")
        return ""

# Function to label sentiment using TextBlob
def label_sentiment_textblob(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 'positive'
    elif polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Function to label sentiment using VADER with adjustable threshold
def label_sentiment_vader_adjusted(text, neutral_threshold=0.1):
    sentiment_score = analyzer.polarity_scores(text)
    compound_score = sentiment_score['compound']
    if compound_score > neutral_threshold:
        return 'positive'
    elif compound_score < -neutral_threshold:
        return 'negative'
    else:
        return 'neutral'

# Streamlit App
st.title("Patient Insight Pro (Inpatient)")

# Define tabs
tabs = ["Overview and Purpose", "Data Upload and Overview", "Data Cleaning and Processing", "Visualization and Sentiment Analysis"]

# Create tab navigation
selected_tab = st.radio("Navigation", tabs)

if selected_tab == "Overview and Purpose":
    st.write("### Leveraging Sentiment Analysis to Enhance Patient Experience and Satisfaction")
    st.write("Healthcare organizations continuously seek ways to improve patient care and satisfaction.")
    st.write("- Efficient data preprocessing and cleaning.")
    st.write("- Sentiment classification using TextBlob and VADER.")
    st.write("- Insights through data visualization.")

elif selected_tab == "Data Upload and Overview":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        data_hc = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(data_hc.head())
        st.write("### Dataset Information")
        st.text(data_hc.info())
        st.write("### Missing Values")
        st.write(data_hc.isnull().sum())

elif selected_tab == "Data Cleaning and Processing":
    if uploaded_file:
        st.write("### Cleaning Dataset")
        columns_to_remove = [
            'Patient Survey Star Rating Footnote',
            'HCAHPS Answer Percent Footnote',
            'Number of Completed Surveys Footnote',
            'Survey Response Rate Percent Footnote'
        ]
        data_hc = data_hc.drop(columns=columns_to_remove, errors='ignore')
        st.write("### Updated Dataset Columns")
        st.write(data_hc.columns)

elif selected_tab == "Visualization and Sentiment Analysis":
    if uploaded_file:
        if 'HCAHPS Answer Description' in data_hc.columns:
            st.write("### Sentiment Analysis")
            data_hc['Cleaned_Answer_Description'] = data_hc['HCAHPS Answer Description'].fillna("").apply(clean_text)
            data_hc['TextBlob_Sentiment'] = data_hc['Cleaned_Answer_Description'].apply(label_sentiment_textblob)
            data_hc['VADER_Sentiment'] = data_hc['Cleaned_Answer_Description'].apply(label_sentiment_vader_adjusted)

            # Visualize sentiment distribution
            st.write("### Sentiment Distribution")
            sentiment_counts = data_hc['VADER_Sentiment'].value_counts()
            st.bar_chart(sentiment_counts)

            # Word Cloud
            st.write("### Word Cloud")
            all_text = ' '.join(data_hc['Cleaned_Answer_Description'])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
        else:
            st.error("Column 'HCAHPS Answer Description' not found in the dataset.")
