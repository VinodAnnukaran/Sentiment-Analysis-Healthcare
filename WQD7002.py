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
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Download necessary NLTK resources
download('punkt')
download('stopwords')
download('wordnet')

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

  
