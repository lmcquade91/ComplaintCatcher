import streamlit as st
import pandas as pd
import openai

st.title("Hotel Review Sentiment Analysis Dashboard")
st.write("Filter and analyze hotel reviews based on sentiment, category, and date.")

# Load data
@st.cache_data
def load_data():
    file_path = "REVIEWS_WITH_LABELS_AND_CATEGORY.pkl"  # Update if necessary
    return pd.read_pickle(file_path)

df = load_data()

# Sidebar filters
st.sidebar.header("Filter Options")

# Category filter
categories = df["Category"].unique().tolist()
selected_category = st.sidebar.selectbox("Select Category", ["All"] + categories)

# Sentiment filter
sentiment_options = ["All", "Positive", "Negative"]
selected_sentiment = st.sidebar.selectbox("Select Sentiment", sentiment_options)

# Date range filter
start_date = st.sidebar.date_input("Start Date", pd.to_datetime(df["Date of Review"]).min())
end_date = st.sidebar.date_input("End Date", pd.to_datetime(df["Date of Review"]).max())

# Convert start_date and end_date to Pandas Timestamps after input
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Apply filters
filtered_df = df.copy()

# Category filter
if selected_category != "All":
    filtered_df = filtered_df[filtered_df["Category"] == selected_category]

# Sentiment filter
if selected_sentiment == "Positive":
    filtered_df = filtered_df[filtered_df["predicted_sentiment"] > 0]
elif selected_sentiment == "Negative":
    filtered_df = filtered_df[filtered_df["predicted_sentiment"] <= 0]

# Date filter
filtered_df = filtered_df[
    (pd.to_datetime(filtered_df["Date of Review"]) >= start_date) &
    (pd.to_datetime(filtered_df["Date of Review"]) <= end_date)
]

# Check if the dataframe is empty
if filtered_df.empty:
    st.write("No data to display.")
else:
    # Display filtered reviews
    st.subheader("Filtered Reviews")
    st.write(filtered_df.reset_index(drop=True))  # Display full dataframe without index
    
    # Extract the reviews text from the filtered dataset
    reviews_text = " ".join(filtered_df['Review'].tolist())

    # Make the AI call to summarize the feedback
    openai.api_key = st.secrets["openai_api_key"]  # Get the OpenAI API key from the secrets

    prompt = f"Please summarize the following hotel reviews in bullet points, highlighting key themes such as service, amenities, or any recurring issues:\n\n{reviews_text}"
    
    # Make the API call to GPT-4 for summarization
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract and display the summary
    summary = response['choices'][0]['message']['content']
    st.subheader("Summary of Feedback")
    st.write(summary)

