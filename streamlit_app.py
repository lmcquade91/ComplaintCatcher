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

# Define the 6 main categories explicitly
main_categories = ["Staff/Service", "Room", "Pool", "Hotel", "Booking", "Food & Beverage", "Miscellaneous"]

# Create a copy to ensure we keep all categories
line_chart_data = df.copy()

# Apply sentiment filter while keeping all categories
if selected_sentiment == "Positive":
    line_chart_data = line_chart_data[line_chart_data["predicted_sentiment"] > 0]
elif selected_sentiment == "Negative":
    line_chart_data = line_chart_data[line_chart_data["predicted_sentiment"] <= 0]

# Apply date filter
line_chart_data = line_chart_data[
    (line_chart_data["Date of Review"] >= start_date) &
    (line_chart_data["Date of Review"] <= end_date)
]

# Ensure all categories always appear, even if they have no data
all_dates = pd.date_range(start=start_date, end=end_date, freq="W")  # Weekly intervals
category_expansion = pd.MultiIndex.from_product([all_dates, main_categories], names=["Date of Review", "Category"])

# Group by Date & Category to calculate the average sentiment score
sentiment_over_time = (
    line_chart_data.groupby([line_chart_data["Date of Review"].dt.to_period("W"), "Category"])
    ["sentiment_score"]
    .mean()
    .reset_index()
)

# Convert 'Date of Review' back to a datetime format for plotting
sentiment_over_time["Date of Review"] = sentiment_over_time["Date of Review"].dt.start_time

# Pivot to ensure all categories exist for all weeks, filling missing values with NaN
sentiment_over_time = sentiment_over_time.pivot(index="Date of Review", columns="Category", values="sentiment_score")

# Reindex with all date-category combinations to ensure missing values don't remove lines
sentiment_over_time = sentiment_over_time.reindex(category_expansion, fill_value=None).reset_index()

# Convert back to long format for Plotly
sentiment_over_time = sentiment_over_time.melt(id_vars=["Date of Review"], var_name="Category", value_name="Average Sentiment Score")

# Check if the DataFrame is empty
if sentiment_over_time.empty:
    st.write("No data available to display.")
else:
    # Create the line chart with all categories
    fig = px.line(sentiment_over_time, 
                  x="Date of Review", 
                  y="Average Sentiment Score", 
                  color="Category",
                  title="Sentiment Score Over Time by Category", 
                  labels={"Average Sentiment Score": "Sentiment Score", "Date of Review": "Date"},
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
