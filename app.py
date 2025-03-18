# Ensure 'Date of Review' is a datetime type
filtered_df["Date of Review"] = pd.to_datetime(filtered_df["Date of Review"], errors="coerce")

# Group by Week and Category, then calculate mean sentiment score
filtered_df["Week"] = filtered_df["Date of Review"].dt.to_period("W").astype(str)  # Convert to week period as a string

weekly_sentiment = (
    filtered_df.groupby(["Week", "Category"], as_index=False)
    .agg({"sentiment_score": "mean"})
)

# Convert 'Week' back to datetime format for proper plotting
weekly_sentiment["Week"] = pd.to_datetime(weekly_sentiment["Week"])

# Ensure all categories appear, even if missing some weeks
all_weeks = pd.date_range(start=start_date, end=end_date, freq="W")
category_expansion = pd.MultiIndex.from_product([all_weeks, selected_categories], names=["Week", "Category"])
weekly_sentiment = weekly_sentiment.set_index(["Week", "Category"]).reindex(category_expansion).reset_index()

# Fill missing sentiment scores with NaN to avoid misleading connections
weekly_sentiment["sentiment_score"] = weekly_sentiment["sentiment_score"].astype(float)

# Create the line chart with distinct colors for each category
if weekly_sentiment.empty:
    st.write("No data available to display.")
else:
    fig = px.line(
        weekly_sentiment, 
        x="Week", 
        y="sentiment_score", 
        color="Category",
        title="Sentiment Score Over Time by Category", 
        labels={"sentiment_score": "Sentiment Score", "Week": "Date"},
        markers=True, 
        line_shape="spline",
        color_discrete_map={
            "Staff/Service": "red",
            "Room": "blue",
            "Pool": "green",
            "Hotel": "purple",
            "Booking": "orange",
            "Food & Beverage": "brown",
            "Miscellaneous": "pink"
        }
    )
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
