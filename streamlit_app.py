import streamlit as st
import pandas as pd
import openai

# ✅ Debugging Step: Check if secrets are loaded correctly
st.write("Loaded Secrets:", st.secrets)  # REMOVE THIS IN PRODUCTION

# ✅ Set OpenAI API Key
try:
    openai.api_key = st.secrets["openai_api_key"]
    st.success("✅ OpenAI API Key Loaded Successfully")  # Debugging feedback
except KeyError:
    st.error("❌ OpenAI API Key NOT FOUND. Please check secrets configuration.")
    st.stop()  # Stop execution if no API key

# 🎯 Title
st.title("Hotel Review Sentiment Analysis Dashboard")
st.write("Filter and analyze hotel reviews based on sentiment, category, and date.")

# 📂 Load Data
@st.cache_data
def load_data():
    file_path = "REVIEWS_WITH_LABELS_AND_CATEGORY.pkl"  # Update if necessary
    return pd.read_pickle(file_path)

df = load_data()

# 🎚 Sidebar Filters
st.sidebar.header("Filter Options")

# 🔍 Category Filter
categories = df["Category"].unique().tolist()
selected_category = st.sidebar.selectbox("Select Category", ["All"] + categories)

# 📊 Sentiment Filter
sentiment_options = ["All", "Positive", "Negative"]
selected_sentiment = st.sidebar.selectbox("Select Sentiment", sentiment_options)

# 📅 Date Range Filter
start_date = st.sidebar.date_input("Start Date", pd.to_datetime(df["Date of Review"]).min())
end_date = st.sidebar.date_input("End Date", pd.to_datetime(df["Date of Review"]).max())

# ✅ Convert Dates
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# 🔍 Apply Filters
filtered_df = df.copy()

if selected_category != "All":
    filtered_df = filtered_df[filtered_df["Category"] == selected_category]

if selected_sentiment == "Positive":
    filtered_df = filtered_df[filtered_df["predicted_sentiment"] > 0]
elif selected_sentiment == "Negative":
    filtered_df = filtered_df[filtered_df["predicted_sentiment"] <= 0]

filtered_df = filtered_df[
    (pd.to_datetime(filtered_df["Date of Review"]) >= start_date) &
    (pd.to_datetime(filtered_df["Date of Review"]) <= end_date)
]

# 📜 Display Filtered Data
if filtered_df.empty:
    st.warning("⚠️ No data to display based on selected filters.")
else:
    st.subheader("Filtered Reviews")
    st.write(filtered_df.reset_index(drop=True))  # Display full dataframe without index

    # 🔍 Extract Review Text for Summarization
    reviews_text = " ".join(filtered_df['Review'].tolist())

    # 🚀 Call OpenAI API for Summary
    st.subheader("Summary of Feedback")
    prompt = f"Summarize the following hotel reviews in bullet points, focusing on key themes (service, amenities, issues):\n\n{reviews_text}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        summary = response['choices'][0]['message']['content']
        st.write(summary)
    except Exception as e:
        st.error(f"⚠️ Error calling OpenAI API: {e}")


