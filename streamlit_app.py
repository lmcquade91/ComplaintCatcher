import streamlit as st
import pandas as pd
import openai
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Apply custom styling
st.markdown(
    """
    <style>
        /* Center the title */
        .title { text-align: center; color: #4B8BBE; }
        /* Sidebar styling */
        [data-testid="stSidebar"] { background-color: #f0f2f6; }
    </style>
    """,
    unsafe_allow_html=True
)

# Load data
@st.cache_data
def load_data():
    file_path = "REVIEWS_WITH_LABELS_AND_CATEGORY.pkl"
    return pd.read_pickle(file_path)

df = load_data()

# Title
st.markdown("<h1 class='title'>📊 Key Insights per Category for Hotel Reviews</h1>", unsafe_allow_html=True)

# Dark mode toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

# Sidebar filters
st.sidebar.header("🔎 Filter Options")

categories = df["Category"].unique().tolist()
selected_category = st.sidebar.selectbox("📂 Select Category", ["All"] + categories)

sentiment_options = ["All", "Positive", "Negative"]
selected_sentiment = st.sidebar.select_slider("📈 Select Sentiment", options=sentiment_options)

start_date = st.sidebar.date_input("📅 Start Date", pd.to_datetime(df["Date of Review"]).min())
end_date = st.sidebar.date_input("📅 End Date", pd.to_datetime(df["Date of Review"]).max())

# Convert to Pandas Timestamps
start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)

# Search bar for reviews
search_query = st.sidebar.text_input("🔍 Search Reviews", "")

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

# Apply search filter
if search_query:
    filtered_df = filtered_df[filtered_df["Review"].str.contains(search_query, case=False, na=False)]

# Display filtered data
if filtered_df.empty:
    st.warning("No data available for selected filters.")
else:
    st.subheader("📄 Filtered Reviews")
    st.dataframe(filtered_df.reset_index(drop=True))

# Sentiment Over Time Chart (Interactive)
st.subheader("📊 Sentiment Score Over Time")

sentiment_over_time = filtered_df.groupby(pd.to_datetime(filtered_df["Date of Review"]).dt.date)["sentiment_score"].mean().reset_index()
fig = px.line(sentiment_over_time, x="Date of Review", y="sentiment_score", title="Sentiment Trend Over Time", markers=True)
st.plotly_chart(fig)

# Generate summary
if st.button("📝 Generate Summary"):
    with st.spinner("Analyzing reviews..."):
        client = openai.OpenAI(api_key=st.secrets["openai_api_key"])
        
        reviews_text = " ".join(filtered_df['Review'].tolist())
        prompt = f"Summarize these hotel reviews in bullet points, highlighting key themes such as service, amenities, and recurring issues:\n\n{reviews_text}"
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
            )
            summary = response.choices[0].message.content
            st.success("Summary Generated!")
            st.markdown(f"📌 **Summary:**\n\n{summary}")

        except openai.RateLimitError:
            st.error("Rate limit exceeded. Please wait and try again.")
        except openai.OpenAIError as e:
            st.error(f"OpenAI API Error: {str(e)}")

# Reviews by Category (Pie Chart)
st.subheader("📊 Reviews by Category")
category_counts = filtered_df["Category"].value_counts()
fig = px.pie(values=category_counts, names=category_counts.index, title="Review Distribution by Category")
st.plotly_chart(fig)

# Footer
st.markdown("---")
st.caption("📊 Built with Streamlit | 💡 Data-Driven Insights")
