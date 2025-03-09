import streamlit as st
import pandas as pd

st.title("Hotel Review Sentiment Analysis Dashboard")
st.write("Welcome! This is a Streamlit app to analyze hotel reviews.")

# Load data
@st.cache_data
def load_data():
    file_path = "REVIEWS_WITH_LABELS_AND_CATEGORY.pkl"  # Update if your file has a different path
    return pd.read_pickle(file_path)

# Load dataset
try:
    df = load_data()
    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Dataset Info")
    st.write(df.describe())

except Exception as e:
    st.error(f"Error loading dataset: {e}")
