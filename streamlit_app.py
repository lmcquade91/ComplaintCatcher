import streamlit as st

st.write("App started successfully!")

import pandas as pd
import openai
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
# @st.cache_data
def load_data():
    file_path = "REVIEWS_WITH_LABELS_AND_CATEGORY.pkl"
    df = pd.read_pickle(file_path)
    df = df.drop(columns=["HUMAN LABEL", "embeddings"], errors="ignore")  # Hide unwanted columns
    return df

df = load_data()

# Title with custom font styling
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>ComplaintCatcher</h1>", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("Filter Options")

categories = df["Category"].unique().tolist()
selected_categories = st.sidebar.multiselect("Select Categories", categories, default=categories)

sentiment_options = ["All", "Positive", "Negative"]
selected_sentiment = st.sidebar.selectbox("Select Sentiment", sentiment_options)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime(df["Date of Review"]).min())
end_date = st.sidebar.date_input("End Date", pd.to_datetime(df["Date of Review"]).max())

# Convert to Pandas Timestamps
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Apply filters
filtered_df = df.copy()
filtered_df = filtered_df[filtered_df["Category"].isin(selected_categories)]

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


    # Generate summary button
    if st.button("Generate Summary"):
        st.write("Generating summary...")
        
        # Load API key from Streamlit secrets
        client = openai.OpenAI(api_key=st.secrets["openai_api_key"])
        
        # Join all reviews into a single text block
        reviews_text = " ".join(filtered_df['Review'].tolist())
        
        # Convert selected categories into a readable string
        if selected_categories:
            if len(selected_categories) == 1:
                category_focus = selected_categories[0]
            else:
                category_focus = ", ".join(selected_categories[:-1]) + " and " + selected_categories[-1]
        else:
            category_focus = "the selected category"

        # Update the prompt to focus only on the selected category
        prompt = f"""You are analyzing hotel guest reviews. 
Summarize the following reviews into 10 concise bullet points, focusing only on feedback related to {category_focus}. 
Ignore unrelated topics such as check-in, dining, or services not listed. 
Highlight specific complaints, compliments, or recurring issues about {category_focus} only.

Reviews:
{reviews_text}
"""

        # Call the OpenAI API
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            summary = response.choices[0].message.content
            st.subheader("Summary")
            st.write(summary)

        except openai.RateLimitError:
            st.error("Rate limit exceeded. Please wait and try again.")
        except openai.OpenAIError as e:
            st.error(f"OpenAI API Error: {str(e)}")


# Pie chart showing reviews by category for the selected filters
st.subheader("Reviews by Category")

# Always show all categories in pie chart while applying date and sentiment filters
pie_data = df.copy()

if selected_sentiment == "Positive":
    pie_data = pie_data[pie_data["predicted_sentiment"] > 0]
elif selected_sentiment == "Negative":
    pie_data = pie_data[pie_data["predicted_sentiment"] <= 0]

pie_data = pie_data[
    (pd.to_datetime(pie_data["Date of Review"]) >= start_date) &
    (pd.to_datetime(pie_data["Date of Review"]) <= end_date)
]

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

# Convert 'Date of Review' to datetime format
df["Date of Review"] = pd.to_datetime(df["Date of Review"], errors="coerce")

# Create 'Week' column that represents the start of each week
df["Week"] = df["Date of Review"].dt.to_period("W").apply(lambda r: r.start_time)

# Ensure all categories are included
all_categories = df["Category"].unique()

# Aggregate sentiment scores by week and category
weekly_sentiment = df.groupby(["Week", "Category"])["sentiment_score"].mean().reset_index()

# Pivot to ensure all categories always appear in the plot
weekly_sentiment_pivot = weekly_sentiment.pivot(index="Week", columns="Category", values="sentiment_score").reset_index()

# Fill missing values with 0 (or NaN if preferred)
weekly_sentiment_pivot = weekly_sentiment_pivot.fillna(0)

# Convert 'Week' column to datetime (fixing the error)
weekly_sentiment_pivot["Week"] = pd.to_datetime(weekly_sentiment_pivot["Week"])

# Streamlit visualization
st.subheader("Sentiment Score Over Time by Category")

# Filter data based on user-selected date range
filtered_weekly_sentiment = weekly_sentiment_pivot[
    (weekly_sentiment_pivot["Week"] >= start_date) & 
    (weekly_sentiment_pivot["Week"] <= end_date)
]

# Plot with Plotly Express
fig = px.line(
    filtered_weekly_sentiment, 
    x="Week", 
    y=all_categories, 
    title="Weekly Sentiment Score by Category",
    labels={"value": "Mean Sentiment Score", "Week": "Date"},
    markers=True
)

# Customize appearance
fig.update_traces(line=dict(width=2))
fig.update_layout(
    legend_title="Category",
    xaxis_title="Date",
    yaxis_title="Sentiment Score",
    template="plotly_white",
)

# Display in Streamlit
st.plotly_chart(fig)
