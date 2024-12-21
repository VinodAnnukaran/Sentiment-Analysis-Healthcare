# -*- coding: utf-8 -*-

# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "HCAHPS_Hospital_2023_1.csv"
data_hc = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data_hc.head())

# Check dataset structure
print("\nDataset Information:")
print(data_hc.info())

# Display the columns before removal
print("Columns before removal:")
print(data_hc.columns)

# Remove the specified columns
columns_to_remove = [
    'Patient Survey Star Rating Footnote',
    'HCAHPS Answer Percent Footnote',
    'Number of Completed Surveys Footnote',
    'Survey Response Rate Percent Footnote'
]

data_hc = data_hc.drop(columns=columns_to_remove)

# Display the columns after removal
print("\nColumns after removal:")
print(data_hc.columns)

# Check for missing values
print("\nMissing Values Count:")
print(data_hc.isnull().sum())

# Check dataset structure
print("\nDataset Information:")
print(data_hc.info())

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

# Display unique values before and after conversion
for column in numeric_columns:
    print(f"\nUnique values in '{column}' after conversion:")
    print(data_hc[column].unique())

# Count of missing values for numeric columns
for column in numeric_columns:
    print(f"\nCount of NaN values in '{column}':")
    print(data_hc[column].isnull().sum())

# Summary statistics of numerical features
print("\nSummary Statistics:")
print(data_hc.describe())

# Categorical variables analysis
categorical_columns = data_hc.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"\nValue Counts for {col}:")
    print(data_hc[col].value_counts())

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

for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(data_hc[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()

# Boxplots for numerical features to detect outliers
for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data_hc[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Correlation heatmap for numeric columns
numeric_data = data_hc.select_dtypes(include=['int64', 'float64']).drop(columns=exclude_columns, errors='ignore')
if numeric_data.empty:
    print("No numeric columns available for correlation analysis.")
else:
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

# Bar plots for categorical variables
for col in categorical_columns:
    plt.figure(figsize=(10, 4))
    sns.countplot(data_hc[col], order=data_hc[col].value_counts().index, palette="viridis")
    plt.title(f"Count Plot of {col}")
    plt.xticks(rotation=45)
    plt.show()

# Analyze relationships between numerical and categorical variables
target_column = "Patient Survey Star Rating"
categorical_col = "Facility Name"

if target_column in numerical_columns and categorical_col in categorical_columns:
    data_filtered = data_hc[(data_hc[target_column].notnull()) & (data_hc[target_column] > 0)]
    filtered_row_count = data_filtered.shape[0]
    print(f"Number of rows after filtering: {filtered_row_count}")

    if filtered_row_count > 0:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=data_filtered[categorical_col], y=data_filtered[target_column], palette="Set2")
        plt.title(f"{target_column} vs {categorical_col}", fontsize=16)
        plt.xticks(rotation=45)
        plt.show()
    else:
        print(f"No valid values greater than 0 in '{target_column}'.")
else:
    print("No numeric or categorical columns available for analysis.")

# Top facilities with survey counts and average ratings
facility_survey_count = data_hc['Facility Name'].value_counts()
top_10_facilities = facility_survey_count.head(10)
print("Top 10 Facilities with the Highest Survey Counts:")
print(top_10_facilities)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_facilities.index, y=top_10_facilities.values, palette="viridis")
plt.title("Top 10 Facilities with the Highest Survey Counts", fontsize=16)
plt.xticks(rotation=45)
plt.xlabel("Facility Name")
plt.ylabel("Survey Count")
plt.show()

facility_avg_rating = data_hc.groupby('Facility Name')['Patient Survey Star Rating'].mean()
top_10_facilities_rating = facility_avg_rating.sort_values(ascending=False).head(10)
print("Top 10 Facilities with the Highest Average Survey Ratings:")
print(top_10_facilities_rating)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_facilities_rating.index, y=top_10_facilities_rating.values, palette="viridis")
plt.title("Top 10 Facilities with the Highest Average Survey Ratings", fontsize=16)
plt.xticks(rotation=45)
plt.xlabel("Facility Name")
plt.ylabel("Average Rating")
plt.show()

# Install necessary libraries
!pip install vaderSentiment wordcloud matplotlib seaborn nltk pandas scikit-learn streamlit

# Import required libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import download
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import streamlit as st

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

# Function to refine sentiment based on rules
def refine_sentiment(row):
    text = row['Cleaned_Answer_Description']
    textblob_sentiment = row['TextBlob_Sentiment']
    vader_sentiment = row['VADER_Sentiment']

    if "never" in text:
        return 'negative'

    if textblob_sentiment != vader_sentiment:
        return vader_sentiment
    return textblob_sentiment

# Assuming `data_hc` is a DataFrame containing the data
data_hc['Cleaned_Answer_Description'] = data_hc['HCAHPS Answer Description'].fillna("").apply(clean_text)
data_hc['TextBlob_Sentiment'] = data_hc['Cleaned_Answer_Description'].apply(label_sentiment_textblob)
data_hc['VADER_Sentiment'] = data_hc['Cleaned_Answer_Description'].apply(label_sentiment_vader_adjusted)

# Add polarity scores for analysis
data_hc['TextBlob_Polarity'] = data_hc['Cleaned_Answer_Description'].apply(lambda x: TextBlob(x).sentiment.polarity)
data_hc['VADER_Compound'] = data_hc['Cleaned_Answer_Description'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Refine sentiment
data_hc['Final_Sentiment'] = data_hc.apply(refine_sentiment, axis=1)
data_hc.to_csv("sentiment_labeled_data.csv", index=False)

# Visualize sentiment distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='VADER_Sentiment', data=data_hc, palette='Set2')
plt.title('Distribution of Sentiments (VADER)', fontsize=16)
plt.xlabel('Sentiment', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

# Generate and display word clouds
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data_hc['Cleaned_Answer_Description']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for All Text', fontsize=16)
plt.show()

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
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

# Streamlit Application
st.title("Vinod's Sentiment Analysis App")
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}!")
