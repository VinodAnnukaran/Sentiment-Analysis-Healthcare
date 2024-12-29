# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import download
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Download necessary NLTK resources
download('punkt')
download('stopwords')
download('wordnet')

# Initialize lemmatizer and VADER analyzer
lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

# Title and Description
st.title("Interactive Healthcare Data Analysis")
st.write("Analyze healthcare data interactively with this app.")

# Load Dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    data_hc = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data_hc.head())

    # Columns before removal
    st.write("### Columns before removal:")
    st.write(data_hc.columns)

    # Remove specific columns
    columns_to_remove = [
        'Patient Survey Star Rating Footnote',
        'HCAHPS Answer Percent Footnote',
        'Number of Completed Surveys Footnote',
        'Survey Response Rate Percent Footnote'
    ]
    data_hc = data_hc.drop(columns=columns_to_remove)

    # Columns after removal
    st.write("### Columns after removal:")
    st.write(data_hc.columns)

    # Check for missing values
    st.write("### Missing Values Count:")
    st.write(data_hc.isnull().sum())

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

    # Summary statistics of numerical features
    st.write("### Summary Statistics:")
    st.write(data_hc.describe())

    # Categorical variables analysis
    categorical_columns = data_hc.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        st.write(f"### Value Counts for {col}:")
        st.write(data_hc[col].value_counts())

    # Visualize numerical distributions
    exclude_columns = [
        "ZIP Code",
        "HCAHPS Answer Percent",
        "HCAHPS Linear Mean Value",
        "Number of Completed Surveys",
        "Survey Response Rate Percent"
    ]
    numerical_columns = [
        col for col in data_hc.select_dtypes(include=['int64', 'float64']).columns
        if col not in exclude_columns
    ]

    for col in numerical_columns:
        st.write(f"### Distribution of {col}")
        fig, ax = plt.subplots()
        sns.histplot(data_hc[col], kde=True, bins=30, ax=ax)
        st.pyplot(fig)

    # Boxplots for numerical features
    for col in numerical_columns:
        st.write(f"### Boxplot of {col}")
        fig, ax = plt.subplots()
        sns.boxplot(x=data_hc[col], ax=ax)
        st.pyplot(fig)

    # Correlation heatmap for numeric columns
    numeric_data = data_hc.select_dtypes(include=['int64', 'float64']).drop(columns=exclude_columns, errors='ignore')
    if not numeric_data.empty:
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Bar plots for categorical variables
    for col in categorical_columns:
        st.write(f"### Count Plot of {col}")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(data_hc[col], order=data_hc[col].value_counts().index, palette="viridis", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Sentiment Analysis
    data_hc['Cleaned_Answer_Description'] = data_hc['HCAHPS Answer Description'].fillna("").apply(clean_text)
    data_hc['TextBlob_Sentiment'] = data_hc['Cleaned_Answer_Description'].apply(label_sentiment_textblob)
    data_hc['VADER_Sentiment'] = data_hc['Cleaned_Answer_Description'].apply(label_sentiment_vader_adjusted)

    # Visualize sentiment distribution
    st.write("### Sentiment Distribution (VADER)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='VADER_Sentiment', data=data_hc, palette='Set2', ax=ax)
    plt.title('Distribution of Sentiments (VADER)', fontsize=16)
    st.pyplot(fig)

    # Generate and display word clouds
    st.write("### Word Cloud")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data_hc['Cleaned_Answer_Description']))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Define functions for text cleaning and sentiment labeling
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = text.lower()  # Normalize text
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize words
    return ' '.join(tokens)

def label_sentiment_textblob(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 'positive'
    elif polarity == 0:
        return 'neutral'
    else:
        return 'negative'

def label_sentiment_vader_adjusted(text, neutral_threshold=0.1):
    sentiment_score = analyzer.polarity_scores(text)
    compound_score = sentiment_score['compound']
    if compound_score > neutral_threshold:
        return 'positive'
    elif compound_score < -neutral_threshold:
        return 'negative'
    else:
        return 'neutral'
