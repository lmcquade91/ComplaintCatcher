import streamlit as st
import pandas as pd

# Assuming `filtered_df` is created somewhere in your code
# For example, this is how you might load the data (you should replace this with your actual code)
# filtered_df = pd.read_pickle('path_to_your_data.pkl')

# Check the type of filtered_df
st.write(f"Type of filtered_df: {type(filtered_df)}")

# Check columns and first few rows of the dataframe
st.write("Columns in filtered_df:", filtered_df.columns)
st.write("First few rows of filtered_df:")
st.write(filtered_df.head())

# Check if the dataframe is empty
if filtered_df.empty:
    st.write("No data to display.")
else:
    st.subheader("Filtered Reviews")
    st.write(filtered_df.reset_index(drop=True))  # Display full dataframe without index

# Dataset Statistics
st.subheader("Dataset Statistics")
st.write(filtered_df.describe())

st.write(filtered_df.describe())
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
# Check if the dataframe is empty
if filtered_df.empty:
    st.write("No data to display.")
else:
    st.subheader("Filtered Reviews")
    st.write(filtered_df.reset_index(drop=True))  # Display full dataframe without index
    
# Dataset Statistics
st.subheader("Dataset Statistics")
st.write(filtered_df.describe())

