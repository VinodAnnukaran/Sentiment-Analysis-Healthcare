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
nltk.download('wordnet')
nltk.download('punkt_tab')


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

# Display the title only on the first tab (About)
if selected_tab == "About":
    # Streamlit App
    st.title("Leverage Sentiment Analysis to enhance patient experience and satisfaction (Inpatient)")

# Placeholder for uploaded file
if 'data_hc' not in st.session_state:
    st.session_state.data_hc = None  # Initialize in session state

###########################################################################

# About Tab
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

###########################################################################

# Data Overview Tab
elif selected_tab == "Dataset Overview":
    st.title("Dataset Overview")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file into the session state variable
        st.session_state.data_hc = pd.read_csv(uploaded_file)

        # Display original data
        #st.write("Original Data:")
        #st.dataframe(st.session_state.data_hc)
    
        # Identify duplicates
        duplicates = st.session_state.data_hc[st.session_state.data_hc.duplicated()]
        num_duplicates = len(duplicates)
    
        # Display number of duplicates and duplicate rows
        #st.write(f"Number of duplicate rows: {num_duplicates}")
        #st.write("Duplicate Rows:")
        #st.dataframe(duplicates)
    
        # Drop duplicates and reset index
        st.session_state.data_hc = st.session_state.data_hc.drop_duplicates().reset_index(drop=True)
    
        # Show the shape of the dataset (number of rows and columns)
        #st.write(f"### Dataset Shape: {st.session_state.data_hc.shape[0]} rows, {st.session_state.data_hc.shape[1]} columns")

        # Display cleaned data
        #st.write("Data after removing duplicates:")
        #st.dataframe(st.session_state.data_hc)

        # Display dataset preview and shape information
        #st.write("### Dataset Preview")
        #st.dataframe(st.session_state.data_hc.head())
        
        # Define the columns to be removed
        columns_to_remove = [
            'Patient Survey Star Rating Footnote',
            'HCAHPS Answer Percent Footnote',
            'Number of Completed Surveys Footnote',
            'Survey Response Rate Percent Footnote'
        ]
        
        # Drop the specified columns, check if they exist in the dataset first
        st.session_state.data_hc = st.session_state.data_hc.drop(columns=[col for col in columns_to_remove if col in st.session_state.data_hc.columns])

        # Display the updated dataset preview
        #st.write("### Updated Dataset Preview:")
        #st.write("Dataset Preview:")
        #st.dataframe(st.session_state.data_hc.head())

        ##############################################################

        # Step 1: Identify records to delete
        if 'HCAHPS Answer Description' in st.session_state.data_hc.columns and 'Patient Survey Star Rating' in st.session_state.data_hc.columns:
            # Filter data for records matching the conditions
            records_to_delete = st.session_state.data_hc[
                (st.session_state.data_hc['HCAHPS Answer Description'] == 'Summary star rating') &
                (st.session_state.data_hc['Patient Survey Star Rating'] == 'Not Available')
            ]
    
            # Extract Facility IDs from the filtered records
            facility_ids_to_delete = records_to_delete['Facility ID'].unique()
    
            # Display the Facility IDs to delete
            #st.write("Facility IDs to delete:")
            #st.write(facility_ids_to_delete)
    
            # Step 2: Delete relevant records
            # Filter out records with the identified Facility IDs
            data_hc_cleaned = st.session_state.data_hc[
                ~st.session_state.data_hc['Facility ID'].isin(facility_ids_to_delete)
            ]

            # Store the cleaned data in session state
            st.session_state.cleaned_data = data_hc_cleaned
    
            # Display the dataset shapes for validation
            #st.write(f"Original dataset shape: {st.session_state.data_hc.shape}")
            #st.write(f"Dataset Overview: {data_hc_cleaned.shape}")

            # Show the shape of the dataset (number of rows and columns)
            st.write(f"### Dataset Shape: {data_hc_cleaned.shape[0]} rows, {data_hc_cleaned.shape[1]} columns")
  
            # Optionally display the cleaned dataset
            st.write("Dataset Preview:")
            st.dataframe(data_hc_cleaned)
        else:
            # Handle missing columns gracefully
            st.error("The required columns 'HCAHPS Answer Description' or 'Patient Survey Star Rating' are missing in the uploaded file. Please upload a valid dataset.")
            st.write("Available columns:")
            st.write(st.session_state.data_hc.columns.tolist())

        ##############################################################

        # Convert to numeric, forcing non-numeric values to NaN
        st.session_state.data_hc['Patient Survey Star Rating'] = pd.to_numeric(st.session_state.data_hc['Patient Survey Star Rating'], errors='coerce')
        st.session_state.data_hc['HCAHPS Answer Percent'] = pd.to_numeric(st.session_state.data_hc['HCAHPS Answer Percent'], errors='coerce')
        st.session_state.data_hc['HCAHPS Linear Mean Value'] = pd.to_numeric(st.session_state.data_hc['HCAHPS Linear Mean Value'], errors='coerce')
        st.session_state.data_hc['Number of Completed Surveys'] = pd.to_numeric(st.session_state.data_hc['Number of Completed Surveys'], errors='coerce')
        st.session_state.data_hc['Survey Response Rate Percent'] = pd.to_numeric(st.session_state.data_hc['Survey Response Rate Percent'], errors='coerce')

        #Distribution of Patient Survey Star Rating
        # Define the columns to exclude
        exclude_columns = [
            "ZIP Code",
            "HCAHPS Answer Percent",
            "HCAHPS Linear Mean Value",
            "Number of Completed Surveys",
            "Survey Response Rate Percent"
        ]
        
        # Select numerical columns and exclude the specified ones
        numerical_columns = [
            col for col in st.session_state.data_hc.select_dtypes(include=['int64', 'float64']).columns
            if col not in exclude_columns
        ]
        
        # Display the distribution for numerical variables
        for col in numerical_columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(st.session_state.data_hc[col], kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            
            # Display the plot in Streamlit
            st.pyplot(plt)
            plt.clf()  # Clear the figure for the next plot

            # Filter data for "Summary star rating"
            filtered_data = st.session_state.data_hc[st.session_state.data_hc['HCAHPS Answer Description'] == 'Summary star rating']
            
            # Group by State and Facility, then calculate the average of 'Patient Survey Star Rating'
            best_facility_statewise = filtered_data.groupby(['State', 'Facility Name'])['Patient Survey Star Rating'].mean().reset_index()
            
            # For each state, find the facility with the highest rating
            best_facility_statewise = best_facility_statewise.dropna(subset=['Patient Survey Star Rating'])
            
            best_facility_per_state = best_facility_statewise.loc[best_facility_statewise.groupby('State')['Patient Survey Star Rating'].idxmax()]
            
            # Create a bar chart
            chart = alt.Chart(best_facility_per_state).mark_bar().encode(
                x=alt.X('Facility Name:N', title='Facility Name', sort='-y'),
                y=alt.Y('Patient Survey Star Rating:Q', title='Patient Survey Star Rating'),
                color='State:N',
                tooltip=['Facility Name:N', 'State:N', 'Patient Survey Star Rating:Q']
            ).properties(
                title='Highest Rated Facility by State'
            ).interactive()
            
            # Display the chart in the Streamlit app
            st.altair_chart(chart, use_container_width=True)
            
###############################################
        
        # Initialize lemmatizer and VADER analyzer
        lemmatizer = WordNetLemmatizer()
        analyzer = SentimentIntensityAnalyzer()
        
        # Function to clean and preprocess text
        def clean_text(text):
            """
            Cleans and preprocesses text by removing HTML tags, URLs, 
            converting to lowercase, removing special characters, 
            and applying tokenization, stopword removal, and lemmatization.
            """
            if not isinstance(text, str):
                text = str(text)
        
            # Remove HTML tags and URLs
            text = re.sub(r'<.*?>', '', text)  # HTML tags
            text = re.sub(r'http\S+|www\S+', '', text)  # URLs
        
            # Normalize text
            text = text.lower()  # Lowercase
            text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
        
            # Tokenize and remove stopwords
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
        
            # Perform lemmatization
            lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
            
            return ' '.join(lemmatized_tokens)
        
        # Function to label sentiment using TextBlob
        def label_sentiment_textblob(text):
            """
            Determines sentiment of text using TextBlob polarity.
            Returns 'positive', 'neutral', or 'negative'.
            """
            polarity = TextBlob(text).sentiment.polarity
            if polarity > 0:
                return 'positive'
            elif polarity == 0:
                return 'neutral'
            else:
                return 'negative'
        
        # Function to label sentiment using VADER with adjustable threshold
        def label_sentiment_vader_adjusted(text, neutral_threshold=0.1):
            """
            Determines sentiment of text using VADER's compound score.
            Returns 'positive', 'neutral', or 'negative' based on threshold.
            """
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
            """
            Refines sentiment decision by considering both TextBlob and VADER results,
            with special rules for specific terms.
            """
            text = row['Cleaned_Answer_Description']
            textblob_sentiment = row['TextBlob_Sentiment']
            vader_sentiment = row['VADER_Sentiment']
        
            # Override sentiment if specific terms are present
            if "never" in text:
                return 'negative'
        
            # Default to VADER if results mismatch
            if textblob_sentiment != vader_sentiment:
                return vader_sentiment
            
            return textblob_sentiment
        
        # Streamlit app
        def main():
            """
            Main function to execute sentiment analysis pipeline within Streamlit app.
            """
            if 'HCAHPS Answer Description' in st.session_state.data_hc.columns:
                # Clean text data
                st.session_state.data_hc['Cleaned_Answer_Description'] = (
                    st.session_state.data_hc['HCAHPS Answer Description']
                    .fillna("")
                    .apply(clean_text)
                )
        
                # Apply TextBlob sentiment labeling
                st.session_state.data_hc['TextBlob_Sentiment'] = (
                    st.session_state.data_hc['Cleaned_Answer_Description']
                    .apply(label_sentiment_textblob)
                )
        
                # Apply VADER sentiment labeling
                st.session_state.data_hc['VADER_Sentiment'] = (
                    st.session_state.data_hc['Cleaned_Answer_Description']
                    .apply(label_sentiment_vader_adjusted)
                )
        
                # Add polarity scores for analysis
                st.session_state.data_hc['TextBlob_Polarity'] = (
                    st.session_state.data_hc['Cleaned_Answer_Description']
                    .apply(lambda x: TextBlob(x).sentiment.polarity)
                )
                st.session_state.data_hc['VADER_Compound'] = (
                    st.session_state.data_hc['Cleaned_Answer_Description']
                    .apply(lambda x: analyzer.polarity_scores(x)['compound'])
                )
        
                # Refine sentiment
                st.session_state.data_hc['Final_Sentiment'] = (
                    st.session_state.data_hc.apply(refine_sentiment, axis=1)
                )

                # Display the cleaned data and results in the app
                st.write("Processed Sentiment Data:")
                st.dataframe(st.session_state.data_hc[['Cleaned_Answer_Description', 'TextBlob_Sentiment', 'VADER_Sentiment', 'Final_Sentiment']])
        
        # Ensures this runs only when executed directly
        if __name__ == "__main__":
            main()
                
###############################################
        
        # Define a function to categorize feedback based on the provided categories
        def categorize_feedback(description):
            description = description.lower()  # Convert to lowercase for easier matching
        
            # Nurse communication categories
            if "nurse always communicated well" in description:
                return "Nurse Communication"
            elif "nurse sometimes never communicated well" in description:
                return "Nurse Communication"
            elif "nurse usually communicated well" in description:
                return "Nurse Communication"
        
            # Nurse treatment categories
            elif "nurse always treated courtesy respect" in description:
                return "Nurse Treatment"
            elif "nurse sometimes never treated courtesy respect" in description:
                return "Nurse Treatment"
            elif "nurse usually treated courtesy respect" in description:
                return "Nurse Treatment"
        
            # Nurse listening categories
            elif "nurse always listened carefully" in description:
                return "Nurse Listening"
            elif "nurse sometimes never listened carefully" in description:
                return "Nurse Listening"
            elif "nurse usually listened carefully" in description:
                return "Nurse Listening"
        
            # Nurse explanation categories
            elif "nurse always explained thing could understand" in description:
                return "Nurse Explanation"
            elif "nurse sometimes never explained thing could understand" in description:
                return "Nurse Explanation"
            elif "nurse usually explained thing could understand" in description:
                return "Nurse Explanation"
        
            # Doctor communication categories
            elif "doctor always communicated well" in description:
                return "Doctor Communication"
            elif "doctor sometimes never communicated well" in description:
                return "Doctor Communication"
            elif "doctor usually communicated well" in description:
                return "Doctor Communication"
        
            # Doctor treatment categories
            elif "doctor always treated courtesy respect" in description:
                return "Doctor Treatment"
            elif "doctor sometimes never treated courtesy respect" in description:
                return "Doctor Treatment"
            elif "doctor usually treated courtesy respect" in description:
                return "Doctor Treatment"
        
            # Doctor listening categories
            elif "doctor always listened carefully" in description:
                return "Doctor Listening"
            elif "doctor sometimes never listened carefully" in description:
                return "Doctor Listening"
            elif "doctor usually listened carefully" in description:
                return "Doctor Listening"
        
            # Doctor explanation categories
            elif "doctor always explained thing could understand" in description:
                return "Doctor Explanation"
            elif "doctor sometimes never explained thing could understand" in description:
                return "Doctor Explanation"
            elif "doctor usually explained thing could understand" in description:
                return "Doctor Explanation"
        
            # Patient help categories
            elif "patient always received help soon wanted" in description:
                return "Staff Responsiveness"
            elif "patient sometimes never received help soon wanted" in description:
                return "Staff Responsiveness"
            elif "patient usually received help soon wanted" in description:
                return "Staff Responsiveness"
        
            # Cleanliness categories
            elif "room always clean" in description:
                return "Hospital Cleanliness"
            elif "room sometimes never clean" in description:
                return "Hospital Cleanliness"
            elif "room usually clean" in description:
                return "Hospital Cleanliness"
        
            # Quietness categories
            elif "always quiet night" in description:
                return "Hospital Ward Quietness"
            elif "sometimes never quiet night" in description:
                return "Hospital Ward Quietness"
            elif "usually quiet night" in description:
                return "Hospital Ward Quietness"
        
            # Hospital rating categories
            elif "patient gave rating lower low" in description:
                return "Hospital Rating"
            elif "patient gave rating medium" in description:
                return "Hospital Rating"
            elif "patient gave rating high" in description:
                return "Hospital Rating"
        
            # Recommend hospital categories
            elif "patient would recommend hospital probably would definitely would recommend" in description:
                return "Recommend Hospital"
            elif "yes patient would definitely recommend hospital" in description:
                return "Recommend Hospital"
            elif "yes patient would probably recommend hospital" in description:
                return "Recommend Hospital"
        
        # Streamlit app starts here
        def main():
            st.title("Feedback Categorization and Visualization")
        
            # Categorize feedback
            st.session_state.data_hc['Feedback Category'] = st.session_state.data_hc['Cleaned_Answer_Description'].apply(categorize_feedback)
    
            # Group by category
            feedback_volume = st.session_state.data_hc['Feedback Category'].value_counts()
    
            # Display results
            st.write("### Volume of Feedback Across Categories")
            st.bar_chart(feedback_volume)
    
            # Visualize using Seaborn
            st.write("### Detailed Feedback Category Distribution")
            plt.figure(figsize=(14, 8))
            sns.barplot(x=feedback_volume.values, y=feedback_volume.index, palette="viridis")
            plt.title("Volume of Feedback Across Categories", fontsize=16)
            plt.xlabel("Volume of Feedback")
            plt.ylabel("Feedback Category")
            plt.tight_layout()
            st.pyplot(plt)


###############################################        
           

    else:
        st.warning("Please upload a CSV file to proceed.")

###########################################################################

# Sentiment Insights Tab
elif selected_tab == "Sentiment Insights":
    st.title("Sentiment Insights")
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

###########################################################################

# Recommendations Tab
elif selected_tab == "Recommendations":
    st.title("Recommendations")
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

###########################################################################

# Help Tab
elif selected_tab == "Help": 

    st.title("User Feedback Form")

    # # Custom CSS to create a box around the entire content
    # st.markdown(
    #     """
    #     <style>
    #      .feedback-box {
    #             border: 2px solid #4CAF50;
    #             border-radius: 10px;
    #             padding: 20px;
    #             background-color: #f9f9f9;
    #             box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    #             max-width: 800px;
    #             margin: 0 auto;
    #             height: auto;
    #             max-height: 600px;
    #             overflow: auto;
    #         }
    #         .stTextInput, .stTextArea {
    #             width: 100%;
    #         }
    #     </style>
    #     """, unsafe_allow_html=True
    # )

    # # Wrap the entire content in the box by applying the 'feedback-box' class
    # st.markdown('<div class="feedback-box">', unsafe_allow_html=True)

    # Emoji and contact information
    emoji = "ℹ️"
    contact_email = "vinodakk@gmail.com"
    
    # Concatenate message
    message = f"{emoji} You can always reach me at {contact_email} should you encounter any technical issues or have any feedback to make improvements to this app."
    
    # Display the message using markdown
    st.markdown(message)

    # Initialize an empty dictionary to store feedback (using session state for persistence)
    if "feedback_history" not in st.session_state:
        st.session_state.feedback_history = []

    def collect_feedback():
        """
        Collects user feedback and stores it in session state.
        """
        name = st.text_input("**Please enter your name:**")
        feedback = st.text_area("**Please provide your feedback:**")
        if st.button("**Submit Feedback**"):
            if name and feedback:
                # Append feedback as a dictionary
                st.session_state.feedback_history.append({"name": name, "feedback": feedback})
                st.success("Thank you! Your feedback has been submitted successfully.")
            elif not name:
                st.warning("Please enter your name.")
            elif not feedback:
                st.warning("Please provide your feedback.")

    def show_feedback_history():
        """
        Displays the collected feedback history from session state, only if checkbox is selected.
        """
        if st.checkbox("**Show Previous Feedback**"):
            st.header("Previous Feedback")
            if st.session_state.feedback_history:
                for i, entry in enumerate(st.session_state.feedback_history, 1):
                    st.write(f"**Feedback {i}:**")
                    st.write(f"- **Name:** {entry['name']}")
                    st.write(f"- **Feedback:** {entry['feedback']}")
                    st.markdown("---")
            else:
                st.info("No previous feedback found.")

    # Collect new feedback and display history if checkbox is selected
    collect_feedback()
    show_feedback_history()

    # Close the div for the box
    st.markdown('</div>', unsafe_allow_html=True)

