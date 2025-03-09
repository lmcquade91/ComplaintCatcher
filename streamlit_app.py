import streamlit as st
import pandas as pd

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

# Hide the index column and display the dataframe with a large height
st.subheader("Filtered Reviews")
st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)  # Display full dataframe

# Adjust the display with extra space for reviews
st.subheader("Dataset Statistics")
st.write(filtered_df.describe())

if selected_sentiment == "Positive":
    filtered_df = filtered_df[filtered_df["predicted_sentiment"] > 0]
elif selected_sentiment == "Negative":
    filtered_df = filtered_df[filtered_df["predicted_sentiment"] <= 0]

# Convert start_date and end_date to Pandas Timestamp
start_date = pd.to_datetime(start_date)  
end_date = pd.to_datetime(end_date)  

# Apply the date filter correctly
filtered_df = filtered_df[
    (pd.to_datetime(filtered_df["Date of Review"]) >= start_date) &
    (pd.to_datetime(filtered_df["Date of Review"]) <= end_date)
]

# Display results
st.subheader("Filtered Reviews")
st.write(filtered_df)

st.subheader("Dataset Statistics")
st.write(filtered_df.describe())

