import streamlit as st
import pandas as pd
import openai

# Load data
@st.cache_data
def load_data():
    file_path = "REVIEWS_WITH_LABELS_AND_CATEGORY.pkl"
    return pd.read_pickle(file_path)

df = load_data()

# Sidebar filters
st.sidebar.header("Filter Options")

categories = df["Category"].unique().tolist()
selected_category = st.sidebar.selectbox("Select Category", ["All"] + categories)

sentiment_options = ["All", "Positive", "Negative"]
selected_sentiment = st.sidebar.selectbox("Select Sentiment", sentiment_options)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime(df["Date of Review"]).min())
end_date = st.sidebar.date_input("End Date", pd.to_datetime(df["Date of Review"]).max())

# Convert to Pandas Timestamps
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Apply filters
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

# Display data
if filtered_df.empty:
    st.write("No data to display.")
else:
    st.subheader("Filtered Reviews")
    st.write(filtered_df.reset_index(drop=True))

    # Generate summary button
    if st.button("Generate Summary"):
        st.write("Generating summary...")
        openai.api_key = st.secrets["openai_api_key"]
        
        reviews_text = " ".join(filtered_df['Review'].tolist())
        prompt = f"Please summarize the following hotel reviews in bullet points, highlighting key themes such as service, amenities, or any recurring issues:\n\n{reviews_text}"
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            summary = response['choices'][0]['message']['content']
            st.subheader("Summary of Feedback")
            st.write(summary)
        except openai.error.RateLimitError:
            st.error("Rate limit exceeded. Try again later.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
