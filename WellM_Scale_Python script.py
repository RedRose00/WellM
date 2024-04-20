#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from sklearn.linear_model import LinearRegression

nltk.download('vader_lexicon')

def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)['compound']
    return score

def predict_mental_health(sentiment_scores, mental_health_ratings):
    X = np.array(sentiment_scores).reshape(-1, 1)
    y = np.array(mental_health_ratings)
    model = LinearRegression().fit(X, y)
    predicted_rating = model.predict(X[-1].reshape(1, -1))[0]
    return predicted_rating

def analyze_responses(responses, mental_health_ratings):
    sentiment_scores = []
    for response in responses:
        score = sentiment_analysis(response)
        sentiment_scores.append(score)
    predicted_rating = predict_mental_health(sentiment_scores, mental_health_ratings)
    return predicted_rating

def main():
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

    responses = []

    print("Welcome to the Mental Health Analysis Tool")
    print("Please answer the following questions descriptively.\n")

    for question in questions:
        response = input(question + " ")
        responses.append(response)

    # Mental health ratings corresponding to each response
    mental_health_ratings = [
        1,  # Mindfulness
        3,  # Healthy
        5,  # Moderate
        7,  # At Risk
        8,  # High Risk
        9,  # Severe
        10, # Critical
        9,  # Emergency
        8,  # Life-Threatening
        10  # Critical Emergency
    ]

    predicted_rating = analyze_responses(responses, mental_health_ratings)

    print("\nPredicted Mental Health Rating:", predicted_rating)
    print("Diagnosis: ", end="")
    if predicted_rating <= 3:
        print("You seem to be in a good state of mindfulness.")
    elif 3 < predicted_rating <= 5:
        print("You may be experiencing some mild stress or anxiety. Consider practicing relaxation techniques.")
    elif 5 < predicted_rating <= 7:
        print("Your responses indicate moderate levels of distress. It might be helpful to talk to a mental health professional.")
    elif 7 < predicted_rating <= 9:
        print("Your responses suggest significant distress and may warrant seeking professional help.")
    else:
        print("Your responses indicate severe distress and potential risk. Please seek immediate psychiatric intervention.")

if __name__ == "__main__":
    main()


# In[ ]:




