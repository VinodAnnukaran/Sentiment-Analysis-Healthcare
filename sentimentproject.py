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

# Download necessary NLTK resources
download('punkt')
download('stopwords')
download('wordnet')

# Initialize lemmatizer and VADER analyzer
lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

# Function to clean and preprocess text
def clean_text(text):
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

# Streamlit Tabs
st.title("Interactive Healthcare Data Analysis")

# Define tabs
tabs = st.tabs([
    "Overview and Purpose", 
    "Data Upload and Overview", 
    "Data Cleaning and Processing", 
    "Visualization and Sentiment Analysis"
])

# Tab 1: Overview and Purpose
with tabs[0]:
    # Inject custom CSS to set background image
    st.markdown("""
        <style>
            .reportview-container .main .block-container {
                background-image: url('https://raw.githubusercontent.com/VinodAnnukaran/sentiment-analysis-healthcare/main/medical-care-service.jpeg');
                background-size: cover;
                background-repeat: no-repeat;
                background-position: center center;
                padding: 20px;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

    st.header("Overview and Purpose")
    
    st.write(
        "### Leveraging Sentiment Analysis to Enhance Patient Experience and Satisfaction"
    )
    st.write(
        "Healthcare organizations continuously seek ways to improve patient care and satisfaction. "
        "One innovative approach is leveraging sentiment analysis to analyze patient feedback and surveys. "
        "By processing and understanding patients' sentiments, healthcare providers can identify areas for improvement "
        "and implement targeted strategies to enhance the overall patient experience."
    )
    st.write(
        "This application enables:"
    )
    st.write("- Efficient data preprocessing and cleaning.")
    st.write("- Sentiment classification using advanced techniques like TextBlob and VADER.")
    st.write("- Insights through data visualization, helping organizations prioritize patient needs.")
    st.write("- Building data-driven strategies to foster a positive healthcare experience.")

# Tab 2: Data Upload and Overview
with tabs[1]:
    st.header("Upload and Preview Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        data_hc = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(data_hc.head())
        st.write("### Dataset Information")
        st.text(data_hc.info())
        st.write("### Missing Values")
        st.write(data_hc.isnull().sum())

# Tab 3: Data Cleaning and Processing
with tabs[2]:
    st.header("Data Cleaning and Processing")
    if uploaded_file:
        # Remove specified columns
        columns_to_remove = [
            'Patient Survey Star Rating Footnote',
            'HCAHPS Answer Percent Footnote',
            'Number of Completed Surveys Footnote',
            'Survey Response Rate Percent Footnote'
        ]
        data_hc = data_hc.drop(columns=columns_to_remove, errors='ignore')
        st.write("Columns after removal:")
        st.write(data_hc.columns)

        # Convert specific columns to numeric
        numeric_columns = [
            'Patient Survey Star Rating',
            'HCAHPS Answer Percent',
            'HCAHPS Linear Mean Value',
            'Number of Completed Surveys',
            'Survey Response Rate Percent'
        ]
        for column in numeric_columns:
            data_hc[column] = pd.to_numeric(data_hc[column], errors='coerce')
        
        st.write("### Cleaned Dataset Preview")
        st.dataframe(data_hc.head())

# Tab 4: Visualization and Sentiment Analysis
with tabs[3]:
    st.header("Visualization and Sentiment Analysis")
    if uploaded_file:
        # Sentiment Analysis
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

        st.write("### Correlation Heatmap")
        numeric_data = data_hc.select_dtypes(include=['int64', 'float64'])
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        st.pyplot(plt)
