import streamlit as st
import pandas as pd
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up OpenAI API key
openai.api_key = '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'

# Load CSV data using the new caching method
@st.cache_data
def load_data():
    df = pd.read_csv('exam_issues.csv')  # Make sure your CSV file is in this format
    return df

# Function to find the most similar question from the CSV
def find_similar_question(user_input, df):
    questions = df['issues'].tolist()

    # Use TF-IDF to find the closest question
    vectorizer = TfidfVectorizer().fit_transform([user_input] + questions)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity between user input and the questions in the CSV
    cosine_sim = cosine_similarity([vectors[0]], vectors[1:])
    
    # Get the index of the most similar question
    similar_idx = cosine_sim.argsort()[0][-1]
    return df.iloc[similar_idx]['solutions']

# Function to get a fallback response from OpenAI
def get_openai_response(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # or use gpt-3.5-turbo if preferred
        messages=[
            {"role": "system", "content": "You are an AI assistant that helps students resolve issues during online exams."},
            {"role": "user", "content": user_input}
        ],
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# Streamlit app UI
st.title("Examplify.AI")

st.write("Facing issues during your exam? Enter your problem and get an instant solution.")

# Load the exam issues CSV data
df = load_data()

# User input
user_issue = st.text_area("Describe your issue", height=150, placeholder="Eg: My screen froze, Excel is not working, etc.")

# Display a button for submission
if st.button("Get Solution"):
    if user_issue:
        with st.spinner('Searching for a solution...'):
            # Find the most similar question from the CSV
            solution = find_similar_question(user_issue, df)
            
            # If no good match is found, fall back to OpenAI
            if not solution:
                solution = get_openai_response(user_issue)
            
            st.success("Here's your solution:")
            st.write(f"**Solution:** {solution}")
    else:
        st.error("Please describe your issue first!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center;'>
    © 2024 Exemplify Inc. All rights reserved.
    </div>
""", unsafe_allow_html=True)
