import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get environment variables
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

import requests
from transformers import pipeline
from gtts import gTTS
from collections import Counter
import os
from keybert import KeyBERT
import io
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import pandas as pd
import seaborn as sns

# Initialize sentiment analysis pipeline and keyword extraction model
sentiment_pipeline = pipeline("sentiment-analysis")
kw_model = KeyBERT()

def fetch_news(company):
    """Fetches news articles using NewsAPI."""
    params = {
        "q": company,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10  # Get top 10 articles
    }
    response = requests.get(NEWS_API_URL, params=params)
    data = response.json()

    articles = []
    if "articles" in data:
        for item in data["articles"]:
            title = item.get("title", "No Title")
            summary = item.get("description", "No Summary")
            link = item.get("url", "#")
            articles.append({
                "title": title,
                "summary": summary,
                "link": link
            })
    
    return articles

def analyze_sentiment(text):
    """Performs sentiment analysis on the given text."""
    result = sentiment_pipeline(text)[0]
    label = result["label"]
    return "Positive" if "POSITIVE" in label else "Negative" if "NEGATIVE" in label else "Neutral"

def extract_keywords(text, num_keywords=3):
    """Extracts key topics from the article summary using KeyBERT."""
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english')
    return [kw[0] for kw in keywords[:num_keywords]]

def comparative_analysis(articles):
    """Compares sentiment distribution and extracts common topics."""
    sentiment_counts = Counter(article["sentiment"] for article in articles)
    topic_sets = [set(article["topics"]) for article in articles if article["topics"]]

    # Find common and unique topics
    common_topics = set.intersection(*topic_sets) if topic_sets else set()
    unique_topics = [
        {"Article": i+1, "Unique Topics": list(topic_sets[i] - common_topics)}
        for i in range(len(topic_sets))
    ]

    return {
        "Sentiment Distribution": dict(sentiment_counts),
        "Overall Sentiment": max(sentiment_counts, key=sentiment_counts.get, default="Neutral"),
        "Topic Analysis": {
            "Common Topics": list(common_topics),
            "Unique Topics Per Article": unique_topics
        }
    }

def generate_summary(sentiment_counts):
    """Generates a high-level summary based on sentiment analysis results."""
    if sentiment_counts.get("Positive", 0) > sentiment_counts.get("Negative", 0):
        return "समाचार कवरेज मुख्य रूप से सकारात्मक है। कंपनी की संभावनाएँ उज्ज्वल दिखती हैं।"
    elif sentiment_counts.get("Negative", 0) > sentiment_counts.get("Positive", 0):
        return "समाचार कवरेज मुख्य रूप से नकारात्मक है। कंपनी को कुछ चुनौतियों का सामना करना पड़ सकता है।"
    else:
        return "समाचार कवरेज संतुलित है, जिसमें सकारात्मक और नकारात्मक दोनों दृष्टिकोण शामिल हैं।"

def text_to_speech(text):
    """Converts text to Hindi speech and saves the output file."""
    tts = gTTS(text, lang="hi")
    file_path = "output.mp3"
    tts.save(file_path)
    return file_path

def generate_detailed_report(articles, company):
    """Generates a comprehensive analysis report of news articles."""
    # Create report data structure
    report = {
        "company": company,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "article_count": len(articles),
        "sentiment_analysis": {},
        "topic_analysis": {},
        "temporal_analysis": {},
        "visualization_paths": []
    }
    
    # Sentiment analysis section
    sentiment_counts = Counter(article["sentiment"] for article in articles)
    report["sentiment_analysis"] = {
        "distribution": dict(sentiment_counts),
        "predominant_sentiment": max(sentiment_counts, key=sentiment_counts.get, default="Neutral"),
        "sentiment_ratio": {
            "positive_ratio": sentiment_counts.get("Positive", 0) / len(articles) if articles else 0,
            "negative_ratio": sentiment_counts.get("Negative", 0) / len(articles) if articles else 0,
            "neutral_ratio": sentiment_counts.get("Neutral", 0) / len(articles) if articles else 0
        }
    }
    
    # Topic analysis
    all_topics = [topic for article in articles for topic in article.get("topics", [])]
    topic_counter = Counter(all_topics)
    report["topic_analysis"] = {
        "top_topics": dict(topic_counter.most_common(5)),
        "topic_count": len(set(all_topics)),
        "topic_frequency": dict(topic_counter)
    }
    
    # Generate visualizations
    report["visualization_paths"].append(plot_sentiment_distribution(sentiment_counts))
    report["visualization_paths"].append(generate_word_cloud(articles, company))
    report["visualization_paths"].append(plot_topic_frequency(topic_counter))
    
    return report

def generate_word_cloud(articles, company):
    """Generates a word cloud from article summaries."""
    text = " ".join([article.get("summary", "") for article in articles])
    wordcloud = WordCloud(width=800, height=400, background_color="white", max_words=100).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud for {company} News")
    filename = f"{company.lower()}_wordcloud.png"
    plt.savefig(filename)
    plt.close()
    return filename

def plot_topic_frequency(topic_counter):
    """Plots the frequency of top topics."""
    top_topics = topic_counter.most_common(10)
    topics, counts = zip(*top_topics) if top_topics else ([], [])
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(topics)), counts, align='center')
    plt.yticks(range(len(topics)), topics)
    plt.xlabel('Frequency')
    plt.title('Top Topics Mentioned')
    filename = "topic_frequency.png"
    plt.savefig(filename)
    plt.close()
    return filename

def export_report_to_json(report, filename=None):
    """Exports the report to a JSON file."""
    if filename is None:
        filename = f"report_{report['company']}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=4)
    
    return filename

def query_articles(articles, query_params):
    """
    Query and filter articles based on various parameters.
    
    Parameters:
    - articles: List of article dictionaries
    - query_params: Dictionary with filtering parameters:
        - sentiment: Filter by sentiment (Positive, Negative, Neutral)
        - topics: Filter by topics
        - keywords: Full-text search in title and summary
        - sort_by: Sorting criteria (relevance, date, sentiment)
        - limit: Maximum number of results to return
    """
    filtered_articles = articles.copy()
    
    # Filter by sentiment
    if "sentiment" in query_params and query_params["sentiment"]:
        filtered_articles = [a for a in filtered_articles if a.get("sentiment") == query_params["sentiment"]]
    
    # Filter by topics
    if "topics" in query_params and query_params["topics"]:
        requested_topics = set(query_params["topics"])
        filtered_articles = [
            a for a in filtered_articles 
            if any(topic in requested_topics for topic in a.get("topics", []))
        ]
    
    # Full-text search in title and summary
    if "keywords" in query_params and query_params["keywords"]:
        keywords = query_params["keywords"].lower()
        filtered_articles = [
            a for a in filtered_articles 
            if keywords in a.get("title", "").lower() or keywords in a.get("summary", "").lower()
        ]
    
    # Apply sorting
    if "sort_by" in query_params:
        if query_params["sort_by"] == "sentiment":
            # Sort by sentiment (Positive first, then Neutral, then Negative)
            sentiment_order = {"Positive": 0, "Neutral": 1, "Negative": 2}
            filtered_articles.sort(key=lambda a: sentiment_order.get(a.get("sentiment", "Neutral"), 1))
    
    # Apply limit
    if "limit" in query_params and query_params["limit"] > 0:
        filtered_articles = filtered_articles[:query_params["limit"]]
    
    return filtered_articles

def extract_entities(text):
    """
    Extract named entities from text (companies, people, locations).
    This is a simplified version. For production, consider using spaCy or other NER tools.
    """
    # This is a placeholder. Ideally, use a proper NER system
    return []

def calculate_relevance_score(article, query):
    """Calculate relevance score of an article to a search query."""
    score = 0
    
    # Title match (higher weight)
    if query.lower() in article.get("title", "").lower():
        score += 3
    
    # Summary match
    if query.lower() in article.get("summary", "").lower():
        score += 1
    
    # Topic match
    if any(query.lower() in topic.lower() for topic in article.get("topics", [])):
        score += 2
    
    return score

def plot_sentiment_distribution(sentiment_counts):
    """Generates a bar chart for sentiment distribution."""
    df = pd.DataFrame(sentiment_counts.items(), columns=["Sentiment", "Count"])
    plt.figure(figsize=(6, 4))
    sns.barplot(x="Sentiment", y="Count", data=df, palette="coolwarm")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    file_path = "sentiment_plot.png"
    plt.savefig(file_path)
    plt.close()
    return file_path
