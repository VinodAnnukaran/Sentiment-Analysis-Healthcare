#Streamlit app to leverage sentiment analysis to enhance patient experience and satisfaction
# Importing Libraries
# Basic Libraries
import pandas as pd
import numpy as np

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from wordcloud import WordCloud

# Text Processing Libraries
import re
import nltk
from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Sentiment Analysis Libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Data Analysis and Modeling Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# Other Utilities
from collections import Counter

# Streamlit Library
import streamlit as st

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Streamlit App
st.title("Leverage Sentiment Analysis to enhance patient experience and satisfaction (Inpatient)")

# Define tabs
tabs = ["About", "Dataset Overview", "Sentiment Insights", "Recommendations","Help"]

# Sidebar navigation
selected_tab = st.sidebar.radio("Navigation", tabs)

# Placeholder for uploaded file
if 'data_hc' not in st.session_state:
    st.session_state.data_hc = None  # Initialize in session state

# Overview and Purpose Tab
if selected_tab == "About":
    st.write("Healthcare organizations continuously seek ways to improve patient care and satisfaction.")
    st.write("- Efficient data preprocessing and cleaning.")
    st.write("- Sentiment classification using TextBlob and VADER.")
    st.write("- Insights through data visualization.")
    st.markdown("<h2>Second Subheading</h2>")
    st.write("Some content under the second subheading.")


# Data Upload and Overview Tab
elif selected_tab == "Dataset Overview":
    st.title("Data Upload and Overview")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Read the file into the session state variable
        st.session_state.data_hc = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(st.session_state.data_hc.head())
        st.write("### Dataset Information")
        st.text(st.session_state.data_hc.info())
        st.write("### Missing Values")
        st.write(st.session_state.data_hc.isnull().sum())
    else:
        st.warning("Please upload a CSV file to proceed.")

# Data Cleaning and Processing Tab
elif selected_tab == "Sentiment Insights":
    st.title("Data Cleaning and Processing")
    if st.session_state.data_hc is not None:
        st.write("### Cleaning Dataset")
        columns_to_remove = [
            'Patient Survey Star Rating Footnote',
            'HCAHPS Answer Percent Footnote',
            'Number of Completed Surveys Footnote',
            'Survey Response Rate Percent Footnote'
        ]
        st.session_state.data_hc = st.session_state.data_hc.drop(columns=columns_to_remove, errors='ignore')
        st.write("### Updated Dataset Columns")
        st.write(st.session_state.data_hc.columns)
    else:
        st.warning("Please upload a CSV file in the 'Data Upload and Overview' tab.")

# Visualization and Sentiment Analysis Tab
elif selected_tab == "Visualization and Sentiment Analysis":
    st.title("Visualization and Sentiment Analysis")
    if st.session_state.data_hc is not None:
        if 'HCAHPS Answer Description' in st.session_state.data_hc.columns:
            st.write("### Sentiment Analysis")
            st.session_state.data_hc['Cleaned_Answer_Description'] = st.session_state.data_hc['HCAHPS Answer Description'].fillna("").apply(clean_text)
            st.session_state.data_hc['TextBlob_Sentiment'] = st.session_state.data_hc['Cleaned_Answer_Description'].apply(label_sentiment_textblob)
            st.session_state.data_hc['VADER_Sentiment'] = st.session_state.data_hc['Cleaned_Answer_Description'].apply(label_sentiment_vader_adjusted)

            # Visualize sentiment distribution
            st.write("### Sentiment Distribution")
            sentiment_counts = st.session_state.data_hc['VADER_Sentiment'].value_counts()
            st.bar_chart(sentiment_counts)

            # Word Cloud
            st.write("### Word Cloud")
            all_text = ' '.join(st.session_state.data_hc['Cleaned_Answer_Description'])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
        else:
            st.error("Column 'HCAHPS Answer Description' not found in the dataset.")
    else:
        st.warning("Please upload a CSV file in the 'Data Upload and Overview' tab.")

# Help Tab
elif selected_tab == "Help":
    st.title("Contact for any help related with the App")

    def collect_feedback():
        """
        Collects user feedback using a simple form.

        Returns:
            dict: A dictionary containing the collected feedback.
        """
        feedback = {}
        feedback['rating'] = st.slider("How would you rate your experience?", 1, 5)
        feedback['comments'] = st.text_area("Please provide any additional comments or suggestions:")

        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback!")
            return feedback
        else:
            return None

    # Collect feedback data
    feedback_data = collect_feedback()

    # Check if feedback_data is not None before proceeding
    if feedback_data:
        with open("feedback.txt", "a") as f:
            f.write(str(feedback_data) + "\n")

