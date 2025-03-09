import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
@st.cache_data
def load_data():
    file_path = "REVIEWS_WITH_LABELS_AND_CATEGORY.pkl"
    return pd.read_pickle(file_path)

df = load_data()

# Title with custom font styling
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Key Insights per Category for Hotel Reviews</h1>", unsafe_allow_html=True)

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

# Adjust sentiment filtering based on the predicted_sentiment for display
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

    # Sentiment over time (Line chart) - use sentiment_score for the graph
    sentiment_over_time = filtered_df.groupby(pd.to_datetime(filtered_df["Date of Review"]).dt.date)["sentiment_score"].mean()
    sentiment_over_time_df = sentiment_over_time.reset_index()

    st.subheader("Sentiment Score Over Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=sentiment_over_time_df, x="Date of Review", y="sentiment_score", ax=ax, color='purple')
    ax.set_title("Average Sentiment Score per Day", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Sentiment Score", fontsize=12)
    st.pyplot(fig)

    # Generate summary button
    if st.button("Generate Summary"):
        st.write("Generating summary...")
        
        # Load API key from Streamlit secrets
        client = openai.OpenAI(api_key=st.secrets["openai_api_key"])
        
        reviews_text = " ".join(filtered_df['Review'].tolist())
        prompt = f"Please summarize the following hotel reviews in bullet points, highlighting key themes such as service, amenities, or any recurring issues:\n\n{reviews_text}"
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            summary = response.choices[0].message.content
            st.write(summary)

        except openai.RateLimitError:
            st.error("Rate limit exceeded. Please wait and try again.")
        except openai.OpenAIError as e:
            st.error(f"OpenAI API Error: {str(e)}")

# Pie chart showing reviews by category for the selected filters
st.subheader("Reviews by Category")

# Filter the data for the selected time period and sentiment
pie_data = filtered_df.copy()

# Count the reviews by category
category_counts = pie_data["Category"].value_counts()

# Create the pie chart
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3", len(category_counts)))
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Set title for the pie chart
ax.set_title("Distribution of Reviews by Category", fontsize=16)

# Display the pie chart
st.pyplot(fig)

# Sentiment over time (Line chart) showing all categories for the selected filters
st.subheader("Sentiment Score Over Time by Category")

# Group by Date and Category, and calculate the mean sentiment score for each
sentiment_over_time_by_category = filtered_df.groupby([pd.to_datetime(filtered_df["Date of Review"]).dt.date, "Category"])["sentiment_score"].mean()
sentiment_over_time_by_category_df = sentiment_over_time_by_category.reset_index()

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Use seaborn to plot sentiment score over time for each category
sns.lineplot(data=sentiment_over_time_by_category_df, x="Date of Review", y="sentiment_score", hue="Category", ax=ax, palette="Set2")

# Set the title and labels
ax.set_title("Average Sentiment Score per Day by Category", fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Sentiment Score", fontsize=12)

# Display the plot
st.pyplot(fig)
