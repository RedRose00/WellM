import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')  # One-time download

def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)['compound']
    return score

def interpret_score(score):
    if score <= 3:
        return "You seem to be in a good state of mindfulness."
    elif 3 < score <= 5:
        return "You may be experiencing some mild stress or anxiety. Consider practicing relaxation techniques."
    elif 5 < score <= 7:
        return "Your responses indicate moderate levels of distress. It might be helpful to talk to a mental health professional."
    else:
        return "**Warning:** Your responses suggest significant distress. Please seek professional help immediately. Here are some resources: [National Suicide Prevention Lifeline](https://suicidepreventionlifeline.org/)"

def analyze_responses(responses):
    sentiment_scores = [sentiment_analysis(response) for response in responses]
    average_score = sum(sentiment_scores) / len(sentiment_scores)
    interpretation = interpret_score(average_score)
    return interpretation

st.title("Mental Health Assessment Tool (Disclaimer: for self-assessment only)")
st.write("Please answer the following questions honestly. All responses are confidential.")

questions = [
    "What brings you the most joy these days?",
    "What challenges do you find yourself facing most often?",
    "Describe a recent situation where you felt truly supported.",
    "What are you most excited about for the future?",
    "Is there anything you feel you're missing out on in life?",
    "How comfortable are you expressing your true self to others?",
    "On a scale of 1-10, how satisfied are you with your current work/activities?",
    "Can you share a recent accomplishment you're proud of?",
    "Do you ever feel overwhelmed by negativity? How do you cope?",
    "In your own words, how would you describe your overall mental well-being?"
]

responses = [st.text_input(question) for question in questions]

if all(response != "" for response in responses):  # Check if all fields are filled
    st.button("Analyze My Responses")

if st.button("Analyze My Responses"):
    interpretation = analyze_responses(responses)
    st.write(f"**Interpretation:** {interpretation}")

    st.write("**Important:**")
    st.write("- This tool is for self-assessment purposes only and cannot replace professional diagnosis.")
    st.write("- If you are concerned about your mental health, please seek help from a qualified mental health professional.")

