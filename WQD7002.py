# -*- coding: utf-8 -*-

# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data_hc.head())

# Check dataset structure
print("\nDataset Information:")
print(data_hc.info())

# Display the columns before removal
print("Columns before removal:")
print(data_hc.columns)
