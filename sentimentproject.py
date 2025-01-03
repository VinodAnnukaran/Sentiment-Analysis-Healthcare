import streamlit as st
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# Helper functions for cleaning and sentiment analysis
def clean_text(text):
    # Clean the text by removing special characters, numbers, and extra spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def label_sentiment_textblob(text):
    # Use TextBlob to classify sentiment
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

def label_sentiment_vader_adjusted(text):
    # Use VADER sentiment analysis for classification
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)['compound']
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'


# Streamlit App
st.title("Patient Insight Pro (Inpatient)")

# Define tabs
tabs = ["Overview and Purpose", "Data Upload and Overview", "Data Cleaning and Processing", "Visualization and Sentiment Analysis"]

# Sidebar navigation
selected_tab = st.sidebar.radio("Navigation", tabs)

# Placeholder for uploaded file
if 'data_hc' not in st.session_state:
    st.session_state.data_hc = None  # Initialize in session state

# Overview and Purpose Tab
if selected_tab == "Overview and Purpose":
    st.title("Overview and Purpose")
    st.write("### Leveraging Sentiment Analysis to Enhance Patient Experience and Satisfaction")
    st.write("Healthcare organizations continuously seek ways to improve patient care and satisfaction.")
    st.write("- Efficient data preprocessing and cleaning.")
    st.write("- Sentiment classification using TextBlob and VADER.")
    st.write("- Insights through data visualization.")

# Data Upload and Overview Tab
elif selected_tab == "Data Upload and Overview":
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
elif selected_tab == "Data Cleaning and Processing":
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
