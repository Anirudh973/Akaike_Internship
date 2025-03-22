import streamlit as st
import requests
import json
from PIL import Image
import io
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
import os
from utils import fetch_news, analyze_sentiment, extract_keywords, comparative_analysis, generate_summary, text_to_speech

# Set page configuration
st.set_page_config(
    page_title="News Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoints
BASE_URL = "http://127.0.0.1:8000"  # Your FastAPI server URL

def get_news(company):
    response = requests.get(f"{BASE_URL}/news/{company}")
    return response.json() if response.status_code == 200 else None

def get_audio(company):
    response = requests.get(f"{BASE_URL}/tts/{company}")
    return response.json() if response.status_code == 200 else None

def get_detailed_report(company):
    response = requests.get(f"{BASE_URL}/news/{company}/report")
    return response.json() if response.status_code == 200 else None

def get_visualization(company, viz_type):
    response = requests.get(f"{BASE_URL}/news/{company}/visualization/{viz_type}", stream=True)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    return None

def query_news(company, query_params):
    response = requests.post(f"{BASE_URL}/news/{company}/query", json=query_params)
    return response.json() if response.status_code == 200 else None

# Display local image
def display_local_image(file_path):
    if os.path.exists(file_path):
        image = Image.open(file_path)
        st.image(image)
    else:
        st.error(f"Image file not found: {file_path}")

# Title and description
st.title("News Sentiment Analyzer")
st.write("Enter a company name to analyze its latest news sentiment.")

# Company name input
company_name = st.text_input("Enter Company Name", value="")

# Initialize or get session state for tabs
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Basic Analysis"

# Create sidebar for feature selection
st.sidebar.title("Features")
feature = st.sidebar.radio(
    "Select Analysis Type",
    ["Basic Analysis", "Detailed Reports", "Query System"]
)

# Button to analyze
if st.button("Analyze"):
    if company_name:
        # Basic news sentiment analysis
        with st.spinner(f"Analyzing news for {company_name}..."):
            news_data = get_news(company_name)
            
            if news_data and "articles" in news_data:
                st.session_state.news_data = news_data
                st.session_state.company_name = company_name
                
                # Get detailed report for visualization
                if feature in ["Detailed Reports", "Query System"]:
                    st.session_state.report_data = get_detailed_report(company_name)
            else:
                st.error(f"No news found for {company_name} or API error.")
    else:
        st.warning("Please enter a company name.")

# Display results based on selected feature
if feature == "Basic Analysis" and 'news_data' in st.session_state:
    st.header("News Articles")
    
    for article in st.session_state.news_data["articles"]:
        st.subheader(article["title"])
        st.write(f"Summary: {article['summary']}")
        st.write(f"Sentiment: {article['sentiment']}")
        st.write(f"Topics: {', '.join(article['topics'])}")
        st.write(f"[Read more]({article['link']})")
        st.markdown("---")
    
    # Display comparative analysis
    st.header("Comparative Analysis")
    st.json(st.session_state.news_data["comparative_analysis"])
    
    # Display summary and audio
    st.header("Final Sentiment Summary")
    st.write(st.session_state.news_data["summary"])
    
    # Get audio for the summary
    audio_data = get_audio(st.session_state.company_name)
    if audio_data and "audio_file" in audio_data:
        st.header("Hindi Text-to-Speech Output")
        st.audio("output.mp3")

elif feature == "Detailed Reports" and 'news_data' in st.session_state:
    st.header(f"Detailed Analysis Report for {st.session_state.company_name}")
    
    if 'report_data' in st.session_state:
        report = st.session_state.report_data
        
        # Display general report information
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Articles", report["report"]["article_count"])
            st.metric("Predominant Sentiment", report["report"]["sentiment_analysis"]["predominant_sentiment"])
        
        with col2:
            sentiment_ratio = report["report"]["sentiment_analysis"]["sentiment_ratio"]
            st.metric("Positive Articles", f"{sentiment_ratio['positive_ratio']:.1%}")
            st.metric("Negative Articles", f"{sentiment_ratio['negative_ratio']:.1%}")
        
        # Display visualizations
        st.subheader("Sentiment Distribution")
        sentiment_viz = get_visualization(st.session_state.company_name, "sentiment")
        if sentiment_viz:
            st.image(sentiment_viz)
        else:
            # Try to display locally
            display_local_image("sentiment_plot.png")
        
        st.subheader("Word Cloud")
        wordcloud_viz = get_visualization(st.session_state.company_name, "wordcloud")
        if wordcloud_viz:
            st.image(wordcloud_viz)
        else:
            # Try to display locally
            display_local_image(f"{st.session_state.company_name.lower()}_wordcloud.png")
        
        st.subheader("Top Topics")
        topics_viz = get_visualization(st.session_state.company_name, "topics")
        if topics_viz:
            st.image(topics_viz)
        else:
            # Try to display locally
            display_local_image("topic_frequency.png")
        
        # Display top topics table
        st.subheader("Top Topics Mentioned")
        if "topic_analysis" in report["report"] and "top_topics" in report["report"]["topic_analysis"]:
            topics_df = pd.DataFrame(
                [(topic, count) for topic, count in report["report"]["topic_analysis"]["top_topics"].items()],
                columns=["Topic", "Frequency"]
            )
            st.table(topics_df)

elif feature == "Query System" and 'news_data' in st.session_state:
    st.header(f"Query System for {st.session_state.company_name} News")
    
    # Set up query parameters
    st.subheader("Filter Articles")
    
    # Create columns for filter options
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment filter
        sentiment_filter = st.selectbox(
            "Filter by Sentiment",
            options=["", "Positive", "Negative", "Neutral"],
            index=0
        )
        
        # Keyword search
        keyword_filter = st.text_input("Search by Keyword")
    
    with col2:
        # Topic filter - get topics from articles
        all_topics = []
        for article in st.session_state.news_data["articles"]:
            all_topics.extend(article.get("topics", []))
        unique_topics = list(set(all_topics))
        
        selected_topics = st.multiselect(
            "Filter by Topics",
            options=unique_topics
        )
        
        # Sorting options
        sort_by = st.selectbox(
            "Sort Results By",
            options=["relevance", "sentiment"],
            index=0
        )
    
    # Results limit
    limit = st.slider("Maximum Results", min_value=1, max_value=20, value=10)
    
    # Execute query button
    if st.button("Apply Filters"):
        query_params = {
            "sentiment": sentiment_filter if sentiment_filter else None,
            "topics": selected_topics if selected_topics else None,
            "keywords": keyword_filter if keyword_filter else None,
            "sort_by": sort_by,
            "limit": limit
        }
        
        query_results = query_news(st.session_state.company_name, query_params)
        
        if query_results and "articles" in query_results:
            st.session_state.query_results = query_results
    
    # Display query results
    if 'query_results' in st.session_state:
        st.subheader(f"Results ({st.session_state.query_results['results_count']} articles)")
        
        for article in st.session_state.query_results["articles"]:
            st.markdown(f"### {article['title']}")
            st.write(f"Summary: {article['summary']}")
            st.write(f"Sentiment: {article['sentiment']}")
            st.write(f"Topics: {', '.join(article['topics'])}")
            st.write(f"[Read more]({article['link']})")
            st.markdown("---")
else:
    st.info("Enter a company name and click 'Analyze' to start.")
