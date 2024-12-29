import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title and Description
st.title("Interactive Healthcare Data Analysis")
st.write("Analyze healthcare data interactively with this app.")

# Load Dataset
st.write("HCAHPS_Hospital_2023_1")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Select column for analysis
    column = st.selectbox("Select a column for analysis:", data.columns)
    if column:
        st.write(f"### Statistics for {column}")
        st.write(data[column].describe())

        st.write(f"### Histogram for {column}")
        fig, ax = plt.subplots()
        data[column].hist(ax=ax, bins=20)
        st.pyplot(fig)
