# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

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

