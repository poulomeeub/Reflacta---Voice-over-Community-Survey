#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import spacy
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

responses=['My overall experience in transportation is good number one the roads are very wide the traffic is very much planned the road signs are visible speculation is very much active therefore I think my overall experience is good',
 "My overall experience in transportation is not good for the roads are sometimes having some damages and a lot of potholes are therefore I don't think it is for good enough experience for me",
 'the overall experience on Transportation is good but there are some issues sometimes in Monsoon and in the snow season there are damages that the road in which is not being repaired at times therefore the Communist face some issues regarding that',
 'I have been driving in Buffalo since 15 years my overall experience is very good',
 "making this very bad it is not good at all the roads have a lot of potholes and the road is unplanned that the road is not white but it's the traffic and there is no traffic police the road signs are not visible"]

# Function to correct spelling
def correct_spelling(text):
    return str(TextBlob(text).correct())

# Function to summarize a single response
def summarize_response(response):
    doc = nlp(response)
    summary = " ".join([sent.text for sent in doc.sents])
    return summary

# Function to analyze sentiment
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Function to extract keywords
def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return keywords

# Streamlit UI
st.title("Response Analysis Dashboard")

if st.button("Analyze Responses"):
    corrected_responses = []
    summaries = []
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    all_keywords = []

    # Process each response
    for i, response in enumerate(responses):
        corrected_response = correct_spelling(response)
        corrected_responses.append(corrected_response)

        summary = summarize_response(corrected_response)
        summaries.append(summary)

        sentiment = analyze_sentiment(corrected_response)
        sentiment_counts[sentiment] += 1

        keywords = extract_keywords(corrected_response)
        all_keywords.extend(keywords)

    # Display sentiment analysis bar chart
    st.subheader("Sentiment Analysis of Responses")
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())
    
    fig, ax = plt.subplots()
    ax.bar(labels, sizes, color=['green', 'red', 'blue'])
    ax.set_title('Sentiment Analysis of Responses')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Number of Responses')
    ax.grid(axis='y')
    st.pyplot(fig)

    # Create and display word cloud
    st.subheader("Word Cloud of Responses")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(corrected_responses))
    
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Display insights
    st.subheader("Insights")
    keyword_counts = Counter(all_keywords)
    most_common_keywords = keyword_counts.most_common(10)

    insights = f"""
    Insights:
    - Positive responses indicate a satisfactory experience, suggesting strengths in the transportation system.
    - Negative feedback points to areas needing improvement, possibly indicating dissatisfaction with specific services or conditions.
    - Common keywords include: {', '.join([kw[0] for kw in most_common_keywords])}.
    """
    st.write(insights)

