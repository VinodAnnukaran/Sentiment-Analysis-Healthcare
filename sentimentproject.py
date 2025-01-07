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
#st.sidebar.markdown('<h2 style="margin-bottom: 0;">Sentilytics PLUS</h2>', unsafe_allow_html=True)
#selected_tab = st.sidebar.radio("", tabs)
st.sidebar.markdown(
    """
    <div style="background-color: #F4F4F4; padding: 10px; border-radius: 10px; text-align: center;">
        <span style="font-size: 24px; font-weight: bold; color: #FF5733;">Sentilytics</span>
        <span style="font-size: 24px; font-weight: bold; color: #4285F4;">PLUS</span>
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
    # Streamlit App with a custom font and black color
    st.markdown(
        """
        <style>
        .custom-title {
            font-family: 'Georgia', serif;
            font-size: 32px;
            font-weight: bold;
            color: black; /* Black color */
            text-align: center;
            margin-top: 20px;
            margin-bottom: 40px; /* Space below the title */
        }
        </style>
        <div class="custom-title">
        Leveraging Machine Learning for Sentiment Analysis of Patient Feedback to Drive Healthcare Improvements (Inpatient)
        </div>
        """,
        unsafe_allow_html=True
    )

# Placeholder for uploaded file
if 'data_hc' not in st.session_state:
    st.session_state.data_hc = None  # Initialize in session state

###########################################################################
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
        - Khaleghparast, S., Maleki, M., Hajianfar, G., Soumari, E., Oveisi, M., Maleki Golandouz, H., Noohi, F., Gholampour Dehaki, M., Golpira, R., Mazloomzadeh, S., Arabian, M., & Kalayinia, S. (2023). Development of a patients’ satisfaction analysis system using machine learning and lexicon-based methods. *BMC Health Services Research, 23(1)*, Article 100. [https://doi.org/10.1186/s12913-023-09260-7](https://doi.org/10.1186/s12913-023-09260-7)
        - Aung, T. T., & Myo, K. (2023). Deep learning-based method for sentiment analysis for patients’ drug reviews. *PeerJ Computer Science, 9*, Article e1976. [https://doi.org/10.7717/peerj-cs.1976](https://doi.org/10.7717/peerj-cs.1976)
        - Giorgi, M., & Sgorbati, S. (2023). Sentiment analysis in healthcare: A methodological review. *Museo Naturalistico, 12(1)*, 1-15.
        - Khanbhai, M., Anyadi, P., Symons, J., Flott, K., Darzi, A., & Mayer, E. (2021). Applying natural language processing and machine learning techniques to patient experience feedback: A systematic review. *BMJ Health & Care Informatics, 28*, e100262. [https://doi.org/10.1136/bmjhci-2020-100262](https://doi.org/10.1136/bmjhci-2020-100262)
        - Sadeghi, A., & Pahlavani, P. (2023). A novel hybrid model for sentiment analysis based on deep learning and ensemble methods. *IEEE Access, 11*, 14567-14580. [https://doi.org/10.1109/ACCESS.2023.10602211](https://doi.org/10.1109/ACCESS.2023.10602211)
    """)


###########################################################################
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

        # Categorize feedback based on description
        def categorize_feedback(description):
            description = description.lower()  # Convert to lowercase for easier matching
            
           # Define the categories with associated keywords
            categories = {
                "Nurse Communication": [
                    "nurse always communicated well", 
                    "nurse sometimes never communicated well", 
                    "nurse usually communicated well"
                ],
                "Nurse Treatment": [
                    "nurse always treated courtesy respect", 
                    "nurse sometimes never treated courtesy respect", 
                    "nurse usually treated courtesy respect"
                ],
                "Nurse Listening": [
                    "nurse always listened carefully", 
                    "nurse sometimes never listened carefully", 
                    "nurse usually listened carefully"
                ],
                "Nurse Explanation": [
                    "nurse always explained thing could understand", 
                    "nurse sometimes never explained thing could understand", 
                    "nurse usually explained thing could understand"
                ],
                "Doctor Communication": [
                    "doctor always communicated well", 
                    "doctor sometimes never communicated well", 
                    "doctor usually communicated well"
                ],
                "Doctor Treatment": [
                    "doctor always treated courtesy respect", 
                    "doctor sometimes never treated courtesy respect", 
                    "doctor usually treated courtesy respect"
                ],
                "Doctor Listening": [
                    "doctor always listened carefully", 
                    "doctor sometimes never listened carefully", 
                    "doctor usually listened carefully"
                ],
                "Doctor Explanation": [
                    "doctor always explained thing could understand", 
                    "doctor sometimes never explained thing could understand", 
                    "doctor usually explained thing could understand"
                ],
                "Staff Responsiveness": [
                    "patient always received help soon wanted", 
                    "patient sometimes never received help soon wanted", 
                    "patient usually received help soon wanted"
                ],
                "Hospital Cleanliness": [
                    "room always clean", 
                    "room sometimes never clean", 
                    "room usually clean"
                ],
                "Hospital Ward Quietness": [
                    "always quiet night", 
                    "sometimes never quiet night", 
                    "usually quiet night"
                ],
                "Hospital Rating": [
                    "patient gave rating lower low", 
                    "patient gave rating medium", 
                    "patient gave rating high"
                ],
                "Hospital Recommendation": [
                    "patient would recommend hospital probably would definitely would recommend", 
                    "yes patient would definitely recommend hospital", 
                    "yes patient would probably recommend hospital"
                ]
            }
                
            for category, phrases in categories.items():
                if any(phrase in description for phrase in phrases):
                    return category
        
            return "Other"  # Return "Other" if no category matches

        # categorize_keywords function
        def categorize_keywords(description):
            description = description.lower()
        
            # Define keywords or phrases for complaints
            complaint_keywords = ['never', 'not', 'problem', 'issue', 'disappointed', 'bad', 'poor', 'worst', 'unhappy']
            compliment_keywords = ['always', 'great', 'good', 'excellent', 'best', 'satisfied', 'wonderful', 'happy', 'love','usually','well']
        
            # Identify complaints or compliments based on keywords
            if any(keyword in description for keyword in complaint_keywords):
                return 'Complaint'
            elif any(keyword in description for keyword in compliment_keywords):
                return 'Compliment'
            else:
                return 'Neutral'
    
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

                # Apply feedback categorization
                st.session_state.data_hc['Feedback Category'] = (
                    st.session_state.data_hc['Cleaned_Answer_Description']
                    .apply(categorize_feedback)
                )

                # Display the cleaned data and results in the app
                #st.write("Processed Sentiment Data:")
                #st.dataframe(st.session_state.data_hc[['Cleaned_Answer_Description', 'TextBlob_Sentiment', 'VADER_Sentiment', 'Final_Sentiment']])

                # Group by category
                #feedback_volume = st.session_state.data_hc['Feedback Category'].value_counts()

                # Unique Feedback Categories
                st.write("### Feedback Categories")
                unique_categories = sorted(st.session_state.data_hc['Feedback Category'].unique())
                #st.write(unique_categories)

                # Display the unique categories in a simple list format
                st.markdown("\n".join([f"- {category}" for category in unique_categories]))
                
                # Visualize feedback volume
                #st.write("### Volume of Feedback Across Categories")
                #st.bar_chart(feedback_volume)
        
                # Detailed Feedback Category Distribution
                #st.write("### Detailed Feedback Category Distribution")
                #plt.figure(figsize=(14, 8))
                #sns.barplot(x=feedback_volume.values, y=feedback_volume.index, palette="viridis")
                #plt.title("Volume of Feedback Across Categories", fontsize=16)
                #plt.xlabel("Volume of Feedback")
                #plt.ylabel("Feedback Category")
                #plt.tight_layout()
                #st.pyplot(plt)

    ###################################
        
                # Apply feedback Keyword
                st.session_state.data_hc['Feedback Keyword'] = (
                    st.session_state.data_hc['Cleaned_Answer_Description']
                    .apply(categorize_keywords)
                )

                # Extract complaints and compliments separately
                complaints = st.session_state.data_hc[st.session_state.data_hc['Feedback Keyword'] == 'Complaint']
                compliments = st.session_state.data_hc[st.session_state.data_hc['Feedback Keyword'] == 'Compliment']
        
                # Get the top complaints and top compliments
                top_complaints = complaints['Cleaned_Answer_Description'].value_counts().head(10).index.tolist()
                top_compliments = compliments['Cleaned_Answer_Description'].value_counts().head(10).index.tolist()

                # Sort the descriptions alphabetically
                top_complaints_sorted = sorted(top_complaints)
                top_compliments_sorted = sorted(top_compliments)
                
                # Display sorted top complaints
                st.write("### Top Complaints")
                for item in top_complaints_sorted:
                    st.markdown(f"<span style='color:red;'>- {item}</span>", unsafe_allow_html=True)

                # Display sorted top compliments
                st.write("### Top Compliments")
                for item in top_compliments_sorted:
                    st.markdown(f"<span style='color:green;'>- {item}</span>", unsafe_allow_html=True)
                
                # Visualize the top complaints and compliments
                #st.write("### Visualizations:")
        
                # Plotting the top complaints and compliments
                #fig, ax = plt.subplots(1, 2, figsize=(14, 8))
        
                # Top Complaints Visualization
                #sns.barplot(x=top_complaints.values, y=top_complaints.index, palette="coolwarm", ax=ax[0])
                #ax[0].set_title("Top Complaints", fontsize=16)
                #ax[0].set_xlabel("Frequency")
                #ax[0].set_ylabel("Complaint Description")
        
                # Top Compliments Visualization
                #sns.barplot(x=top_compliments.values, y=top_compliments.index, palette="viridis", ax=ax[1])
                #ax[1].set_title("Top Compliments", fontsize=16)
                #ax[1].set_xlabel("Frequency")
                #ax[1].set_ylabel("Compliment Description")
        
                # Display the plot in Streamlit
                #st.pyplot(fig)

    ###################################
        
        # Ensures this runs only when executed directly
        if __name__ == "__main__":
            main()
                
    else:
        st.warning("Please upload a CSV file in the 'Dataset Overview' tab.")


###########################################################################
# SENTIMENT INSIGHTS
###########################################################################

# Sentiment Insights Tab
elif selected_tab == "Sentiment Insights":
    st.title("Sentiment Insights")
    if st.session_state.data_hc is not None:
    
###########################
    
        # Preprocessing function to tokenize and remove stopwords
        def preprocess_text(text):
            stop_words = set(stopwords.words('english'))
            exclude_keywords = {'patient', 'nurse', 'hospital', 'doctor'}  # Keywords to exclude
            tokens = word_tokenize(text.lower())
            return [word for word in tokens if word.isalpha() and word not in stop_words and word not in exclude_keywords]        
        
        filtered_data = st.session_state.data_hc[~st.session_state.data_hc['HCAHPS Answer Description'].str.contains('linear mean score|star rating', case=False, na=False)]
        
        # Display sentiment distribution (Final Sentiment)
        st.write("### Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='Final_Sentiment', data=filtered_data, palette='Set2', ax=ax)
        
        # Add data labels
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}',
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha='center', va='center', fontsize=12, color='black', xytext=(0, 8), textcoords='offset points')
        
        # Title and labels
        ax.set_title('Distribution of Sentiments', fontsize=16)
        ax.set_xlabel('Sentiment', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        st.pyplot(fig)
        
        # Helper function to get keywords and plot
        def plot_top_keywords(text, title, palette):
            words = preprocess_text(text)
            top_keywords = Counter(words).most_common(10)
            df = pd.DataFrame(top_keywords, columns=['Keyword', 'Count'])
        
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Count', y='Keyword', data=df, palette=palette, ax=ax)
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Count', fontsize=14)
            ax.set_ylabel('Keyword', fontsize=14)
            st.pyplot(fig)
        
        # Extract text for each sentiment
        positive_text = ' '.join(filtered_data[filtered_data['Final_Sentiment'] == 'positive']['Cleaned_Answer_Description'])
        negative_text = ' '.join(filtered_data[filtered_data['Final_Sentiment'] == 'negative']['Cleaned_Answer_Description'])
        neutral_text = ' '.join(filtered_data[filtered_data['Final_Sentiment'] == 'neutral']['Cleaned_Answer_Description'])
        
        # Plot top keywords
        st.write("### Top Keywords by Sentiment")
        plot_top_keywords(positive_text, 'Top 10 Positive Keywords', 'Greens_r')
        plot_top_keywords(negative_text, 'Top 10 Negative Keywords', 'Reds_r')
        
        # Word cloud generation function
        def generate_wordcloud(text, title):
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(title, fontsize=16)
            ax.axis('off')
            st.pyplot(fig)
        
        # Generate word clouds for all sentiments
        st.write("### Word Clouds")
        generate_wordcloud(' '.join(filtered_data['Cleaned_Answer_Description']), 'Word Cloud for All Text')
        generate_wordcloud(positive_text, 'Word Cloud for Positive Sentiment')
        generate_wordcloud(negative_text, 'Word Cloud for Negative Sentiment')
        generate_wordcloud(neutral_text, 'Word Cloud for Neutral Sentiment')
    
    else:
            st.warning("Please upload a CSV file in the 'Dataset Overview' tab.")
    
    ###########################################################################
    
    # Recommendations Tab
elif selected_tab == "Recommendations":
    st.title("Recommendations")

    if st.session_state.data_hc is not None:
        # Function to categorize feedback based on description
        def categorize_feedback(description):
            description = description.lower()  # Normalize text to lowercase
            categories = {
                "Nurse Communication": [
                    r"nurse always communicated well", 
                    r"nurse sometimes never communicated well", 
                    r"nurse usually communicated well"
                ],
                "Nurse Treatment": [
                    r"nurse always treated courtesy respect", 
                    r"nurse sometimes never treated courtesy respect", 
                    r"nurse usually treated courtesy respect"
                ],
                "Nurse Listening": [
                    r"nurse always listened carefully", 
                    r"nurse sometimes never listened carefully", 
                    r"nurse usually listened carefully"
                ],
                "Nurse Explanation": [
                    r"nurse always explained.*understand", 
                    r"nurse sometimes never explained.*understand", 
                    r"nurse usually explained.*understand"
                ],
                "Doctor Communication": [
                    r"doctor always communicated well", 
                    r"doctor sometimes never communicated well", 
                    r"doctor usually communicated well"
                ],
                "Doctor Treatment": [
                    r"doctor always treated courtesy respect", 
                    r"doctor sometimes never treated courtesy respect", 
                    r"doctor usually treated courtesy respect"
                ],
                "Doctor Listening": [
                    r"doctor always listened carefully", 
                    r"doctor sometimes never listened carefully", 
                    r"doctor usually listened carefully"
                ],
                "Doctor Explanation": [
                    r"doctor always explained.*understand", 
                    r"doctor sometimes never explained.*understand", 
                    r"doctor usually explained.*understand"
                ],
                "Staff Responsiveness": [
                    r"patient always received help soon wanted", 
                    r"patient sometimes never received help soon wanted", 
                    r"patient usually received help soon wanted"
                ],
                "Hospital Cleanliness": [
                    r"room always clean", 
                    r"room sometimes never clean", 
                    r"room usually clean"
                ],
                "Hospital Ward Quietness": [
                    r"always quiet night", 
                    r"sometimes never quiet night", 
                    r"usually quiet night"
                ],
                "Hospital Rating": [
                    r"patient gave rating.*low", 
                    r"patient gave rating.*medium", 
                    r"patient gave rating.*high"
                ],
                "Hospital Recommendation": [
                    r"patient would recommend hospital.*would recommend", 
                    r"yes patient would definitely recommend hospital", 
                    r"yes patient would probably recommend hospital"
                ]
            }

            for category, patterns in categories.items():
                if any(re.search(pattern, description) for pattern in patterns):
                    return category
            return "Other"  # Default category for unmatched descriptions

        # Function to generate recommendations based on sentiment and category
        def generate_recommendation(facility_name, sentiment, category):
            if sentiment == 'Positive':
                return f"Great job, {facility_name}! Continue excelling in {category}. Keep up the good work!"
            elif sentiment == 'Negative':
                return f"{facility_name}, improvements are needed in {category}. Address these concerns to enhance patient satisfaction."
            elif sentiment == 'Neutral':
                return f"{facility_name}, {category} feedback is neutral. Consider further feedback to identify improvement opportunities."
            else:
                return f"{facility_name}, feedback on {category} is mixed. Investigate further to improve patient experience."

        # Facility selection dropdown
        facility_name = st.selectbox("Select Facility", st.session_state.data_hc['Facility Name'].unique())

        # Filter data for the selected facility
        facility_data = st.session_state.data_hc[st.session_state.data_hc['Facility Name'] == facility_name]

        # Exclude specific descriptions
        filtered_data = facility_data[
            ~facility_data['Cleaned_Answer_Description'].str.lower().isin(["linear mean score", "star rating"])
        ]

        # Categorize feedback
        filtered_data['Feedback_Category'] = filtered_data['Cleaned_Answer_Description'].apply(categorize_feedback)

        # Display feedback for the selected facility
        st.write(f"Feedback for {facility_name}:")
        st.dataframe(filtered_data[['Feedback_Category', 'HCAHPS Answer Description', 'Patient Survey Star Rating', 'Final_Sentiment']])

        # Feedback selection dropdown
        selected_feedback = st.selectbox("Select Feedback for Recommendation", filtered_data['HCAHPS Answer Description'].unique())

        # Generate recommendation for the selected feedback
        feedback_data = filtered_data[filtered_data['HCAHPS Answer Description'] == selected_feedback].copy()
        feedback_data['Recommendation'] = feedback_data.apply(
            lambda row: generate_recommendation(facility_name, row['Final_Sentiment'], row['Feedback_Category']), axis=1
        )

        # Display recommendation
        st.write("Recommendation for the selected feedback:")
        st.dataframe(feedback_data[['Recommendation']])
    else:
        st.warning("Please upload a CSV file in the 'Dataset Overview' tab.")

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

