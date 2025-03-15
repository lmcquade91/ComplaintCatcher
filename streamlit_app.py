import streamlit as st
import pandas as pd
import openai
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
@st.cache_data
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

# Ensure 'Date of Review' is a datetime type
df["Date of Review"] = pd.to_datetime(df["Date of Review"], errors="coerce")

# Filter by selected date range (but keep all categories and sentiments)
filtered_df = df[
    (df["Date of Review"] >= start_date) &
    (df["Date of Review"] <= end_date)
]

# Group by date, category, and sentiment category
sentiment_over_time = filtered_df.groupby(
    [filtered_df["Date of Review"].dt.to_period("W"), "Category", "predicted_sentiment"]
)["sentiment_score"].mean().reset_index()

# Convert period to datetime for plotting
sentiment_over_time["Date of Review"] = sentiment_over_time["Date of Review"].dt.start_time

# Map numerical sentiment values to readable labels
sentiment_labels = {0: "Negative", 2: "Positive"}
sentiment_over_time["predicted_sentiment"] = sentiment_over_time["predicted_sentiment"].map(sentiment_labels)

# Check if the DataFrame is empty
if sentiment_over_time.empty:
    st.write("No data available to display.")
else:
    # Create the line chart with sentiment categories separated by color and hotel categories as different lines
    fig = px.line(sentiment_over_time, 
                  x="Date of Review", 
                  y="sentiment_score", 
                  color="predicted_sentiment",
                  line_group="Category",
                  title="Sentiment Score Over Time by Category", 
                  labels={"sentiment_score": "Average Sentiment Score", 
                          "Date of Review": "Date",
                          "predicted_sentiment": "Sentiment",
                          "Category": "Hotel Category"},
                  markers=True, 
                  line_shape="spline")

    # Display the plot
    st.plotly_chart(fig)


    # Generate summary button
    if st.button("Generate Summary"):
        st.write("Generating summary...")
        
        # Load API key from Streamlit secrets
        client = openai.OpenAI(api_key=st.secrets["openai_api_key"])
        
        reviews_text = " ".join(filtered_df['Review'].tolist())
        prompt = f"Please summarize the following hotel reviews into ten bullet points, highlighting key themes such as service, amenities, or any recurring issues:\n\n{reviews_text}"
        
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
