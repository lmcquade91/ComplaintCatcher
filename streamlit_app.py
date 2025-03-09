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
    # Display results
    st.subheader("Filtered Reviews")
    st.write(filtered_df.reset_index(drop=True))  # Display full dataframe without index

    st.subheader("Dataset Statistics")
    st.write(filtered_df.describe())

