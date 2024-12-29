# -*- coding: utf-8 -*-

# Importing Libraries
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
st.write("HCAHPS_Hospital_2023_1")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    data_hc = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data_hc.head())

    # Remove the specified columns
    columns_to_remove = [
        'Patient Survey Star Rating Footnote',
        'HCAHPS Answer Percent Footnote',
        'Number of Completed Surveys Footnote',
        'Survey Response Rate Percent Footnote'
    ]
    data_hc = data_hc.drop(columns=columns_to_remove)

    # Check for missing values
    st.write("\nMissing Values Count:")
    st.write(data_hc.isnull().sum())

    # Conversion of specific columns to numeric
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
    st.write("\nSummary Statistics:")
    st.write(data_hc.describe())

    # Categorical variables analysis
    categorical_columns = data_hc.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        st.write(f"\nValue Counts for {col}:")
        st.write(data_hc[col].value_counts())

    # Define columns to exclude and plot numerical distributions
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

    # Plot numerical distributions
    for col in numerical_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(data_hc[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        st.pyplot()

    # Boxplots for numerical features to detect outliers
    for col in numerical_columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=data_hc[col])
        plt.title(f"Boxplot of {col}")
        st.pyplot()

    # Correlation heatmap for numeric columns
    numeric_data = data_hc.select_dtypes(include=['int64', 'float64']).drop(columns=exclude_columns, errors='ignore')
    if not numeric_data.empty:
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        st.pyplot()

    # Bar plots for categorical variables
    for col in categorical_columns:
        plt.figure(figsize=(10, 4))
        sns.countplot(data_hc[col], order=data_hc[col].value_counts().index, palette="viridis")
        plt.title(f"Count Plot of {col}")
        plt.xticks(rotation=45)
        st.pyplot()

    # Text Preprocessing Functions
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

    # Apply sentiment analysis
    data_hc['Cleaned_Answer_Description'] = data_hc['HCAHPS Answer Description'].fillna("").apply(clean_text)
    data_hc['TextBlob_Sentiment'] = data_hc['Cleaned_Answer_Description'].apply(label_sentiment_textblob)
    data_hc['VADER_Sentiment'] = data_hc['Cleaned_Answer_Description'].apply(label_sentiment_vader_adjusted)

    # Visualize sentiment distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='VADER_Sentiment', data=data_hc, palette='Set2')
    plt.title('Distribution of Sentiments (VADER)', fontsize=16)
    plt.xlabel('Sentiment', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    st.pyplot()

    # Generate and display word clouds
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data_hc['Cleaned_Answer_Description']))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for All Text', fontsize=16)
    st.pyplot()

    # Prepare training and testing sets
    X = data_hc['Cleaned_Answer_Description']
    y = data_hc['VADER_Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorization using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # SVM Model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train_vec, y_train)
    y_pred_svm = svm_model.predict(X_test_vec)
    
    # Show classification report
    st.write("SVM Classification Report:")
    st.text(classification_report(y_test, y_pred_svm))
