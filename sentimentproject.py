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
#st.sidebar.markdown('<h2 style="margin-bottom: 0;">Sentilytics PRO</h2>', unsafe_allow_html=True)
#selected_tab = st.sidebar.radio("", tabs)
st.sidebar.markdown(
    """
    <div style="background-color: #F4F4F4; padding: 10px; border-radius: 10px; text-align: center;">
        <span style="font-size: 24px; font-weight: bold; color: #FF5733;">Sentilytics</span>
        <span style="font-size: 24px; font-weight: bold; color: #4285F4;">PRO</span>
    </div>
    """, unsafe_allow_html=True
)

# Apply custom CSS for styling
st.markdown(
    """
    <style>
        /* Make sidebar background gray */
        [data-testid="stSidebar"] {
            background-color: #E6F0FF; /* Light blue */
        }

        /* Make sidebar tabs bold */
        [data-testid="stSidebar"] .css-1n76uvr {
            font-weight: bold !important;
        }
    </style>
    """, 
    unsafe_allow_html=True
)

# Sidebar radio button
selected_tab = st.sidebar.radio("", ["About", "Dataset Overview", "Sentiment Insights", "Recommendations", "Help"])

# Display content for the selected tab
if selected_tab == "About":
    st.write("Welcome to the **About** section! Here we describe the purpose of Sentilytics PLUS.")
elif selected_tab == "Dataset Overview":
    st.write("Explore the **Dataset Overview** here. This section provides data details and statistics.")
elif selected_tab == "Sentiment Insights":
    st.write("Dive into **Sentiment Insights**, analyzing patient feedback trends and scores.")
elif selected_tab == "Recommendations":
    st.write("In **Recommendations**, discover actionable strategies based on sentiment analysis.")
elif selected_tab == "Help":
    st.write("Visit the **Help** section for FAQs, tutorials, and support resources.")


# Placeholder for uploaded file
if 'data_hc' not in st.session_state:
    st.session_state.data_hc = None  # Initialize in session state

# Overview and Purpose Tab
if selected_tab == "About":
    st.subheader("Background")
    st.write("Patients’ rights are integral to medical ethics. This study aimed to perform sentiment analysis on patients’ feedback by machine learning method to identify positive, negative or neutral comments and to provide recommendation to enhance patient experience.")
    st.subheader("Limitations of Existing Research")
    st.write("Current sentiment analysis tools also face significant limitations, such as: Potential Inaccuracies: Sentiment analysis tools often struggle with detecting nuances in language, such as sarcasm, irony, or idiomatic expressions, leading to misclassification of sentiments.")
    st.subheader("Data Used")
    st.write("We have used a year 2023 dataset from Hospital Consumer Assessment of Healthcare Providers and Systems (HCAHPS). HCAHPS is a national, standardized survey of hospital patients about their experiences during a recent inpatient hospital stay")
    st.write("There are 22 columns in this dataset and 444447 rows but for this analysis we have considered 50,000 records. https://data.cms.gov/provider-data/dataset/dgck-syfz")
    st.subheader("Model Performance")
    st.write("Some content under the first subheading.")
    st.subheader("References")
    st.write("""
        - Zhang, Y., & Xu, H. (2024). A deep learning approach for sentiment analysis in healthcare: A case study on patient feedback. *Procedia Computer Science, 205*, 123-130. [https://doi.org/10.1016/j.procs.2024.01.3139](https://doi.org/10.1016/j.procs.2024.01.3139)
        - Mina, R., & Bahaa, I. (2024). User satisfaction with Arabic COVID-19 apps: Sentiment analysis of users’ reviews using machine learning techniques. *Computers in Human Behavior, 145*, Article 107760. [https://doi.org/10.1016/j.chb.2024.107760](https://doi.org/10.1016/j.chb.2024.107760)
        - Srisankar, M., & Lochanambal, K. P. (2024). A survey on sentiment analysis techniques in the medical domain. *Medicon Agriculture & Environmental Sciences, 6(2)*, 4-9. [https://doi.org/10.55162/MCAES.06.157](https://doi.org/10.55162/MCAES.06.157)
        - Huang, J., Li, X., Chen, Y., & Wang, Y. (2024). The impact of visual function on quality of life in patients with diabetic retinopathy. *Optometry and Vision Science, 101(6)*, 123-130. [https://doi.org/10.1097/OPX.00000000000000013](https://doi.org/10.1097/OPX.00000000000000013)
        - Statista. (2023). Number of public and private hospitals in Malaysia from 2017 to 2022. [https://www.statista.com/statistics/794860/number-of-public-and-private-hospitals-malaysia](https://www.statista.com/statistics/794860/number-of-public-and-private-hospitals-malaysia)
        """)
  


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
    st.title("User Feedback Form")

    # Emoji and contact information
    emoji = "ℹ️"
    contact_email = "vinodakk@gmail.com"
    
    # Concatenate message
    message = f"{emoji} You can always reach me at {contact_email} should you encounter any technical issues or have any feedback to make improvements to this app."
    
    # Display the message using markdown
    st.markdown(message)

    # Initialize an empty list to store feedback
    feedback_history = []
    
    def collect_feedback():
        """
        Collects user feedback and stores it in the history.
        """
        feedback = st.text_area("Please provide your feedback:")
        if st.button("Submit Feedback"):
            if feedback:
                feedback_history.append(feedback)
                st.success("Feedback submitted successfully!")
            else:
                st.warning("Please enter some feedback.")
    
    def show_feedback_history():
        """
        Displays the collected feedback history.
        """
        st.header("Previous Feedback")
        if feedback_history:
            for i, feedback in enumerate(feedback_history):
                st.write(f"**Feedback {i+1}:** {feedback}")
        else:
            st.info("No previous feedback found.")
             
    collect_feedback()
    show_feedback_history()

