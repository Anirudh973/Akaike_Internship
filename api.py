from fastapi import FastAPI, Query, Path, Body
from typing import List, Optional
from pydantic import BaseModel
from utils import (
    fetch_news, analyze_sentiment, extract_keywords, comparative_analysis, 
    generate_summary, text_to_speech, generate_detailed_report, 
    export_report_to_json, query_articles, plot_sentiment_distribution, generate_word_cloud, plot_topic_frequency
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from collections import Counter

app = FastAPI()

class QueryParams(BaseModel):
    sentiment: Optional[str] = None
    topics: Optional[List[str]] = None
    keywords: Optional[str] = None
    sort_by: Optional[str] = "relevance"
    limit: Optional[int] = 10

# Mount static files directory for serving visualizations
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/news/{company}")
def get_news(company: str):
    """Fetches news articles and performs sentiment analysis."""
    articles = fetch_news(company)
    for article in articles:
        article["sentiment"] = analyze_sentiment(article["summary"])
        article["topics"] = extract_keywords(article["summary"])

    analysis = comparative_analysis(articles)
    summary_text = generate_summary(analysis["Sentiment Distribution"])

    return {
        "company": company,
        "articles": articles,
        "comparative_analysis": analysis,
        "summary": summary_text
    }

@app.get("/tts/{company}")
def get_audio(company: str):
    """Generates a Hindi audio summary based on sentiment analysis."""
    articles = fetch_news(company)
    for article in articles:
        article["sentiment"] = analyze_sentiment(article["summary"])
        article["topics"] = extract_keywords(article["summary"])

    analysis = comparative_analysis(articles)
    summary_text = generate_summary(analysis["Sentiment Distribution"])

    audio_file = text_to_speech(summary_text)

    return {
        "summary_text": summary_text,
        "audio_file": audio_file
    }

@app.get("/news/{company}/report")
def get_detailed_report(company: str):
    """Generates a detailed analysis report for the company news."""
    articles = fetch_news(company)
    for article in articles:
        article["sentiment"] = analyze_sentiment(article["summary"])
        article["topics"] = extract_keywords(article["summary"])
    
    report = generate_detailed_report(articles, company)
    report_file = export_report_to_json(report)
    
    return {
        "company": company,
        "report": report,
        "report_file": report_file,
        "visualizations": [f"/static/{path}" for path in report["visualization_paths"]]
    }

@app.post("/news/{company}/query")
def query_news(company: str, query_params: QueryParams):
    """Queries news articles based on provided parameters."""
    articles = fetch_news(company)
    for article in articles:
        article["sentiment"] = analyze_sentiment(article["summary"])
        article["topics"] = extract_keywords(article["summary"])
    
    filtered_articles = query_articles(
        articles, 
        {
            "sentiment": query_params.sentiment,
            "topics": query_params.topics,
            "keywords": query_params.keywords,
            "sort_by": query_params.sort_by,
            "limit": query_params.limit
        }
    )
    
    return {
        "company": company,
        "query_params": query_params.dict(),
        "results_count": len(filtered_articles),
        "articles": filtered_articles
    }

@app.get("/news/{company}/visualization/{viz_type}")
def get_visualization(company: str, viz_type: str = Path(..., description="Type of visualization to generate")):
    """Returns a specific visualization for the company news."""
    articles = fetch_news(company)
    for article in articles:
        article["sentiment"] = analyze_sentiment(article["summary"])
        article["topics"] = extract_keywords(article["summary"])
    
    if viz_type == "sentiment":
        sentiment_counts = Counter(article["sentiment"] for article in articles)
        file_path = plot_sentiment_distribution(sentiment_counts)
        return FileResponse(file_path)
    elif viz_type == "wordcloud":
        file_path = generate_word_cloud(articles, company)
        return FileResponse(file_path)
    elif viz_type == "topics":
        all_topics = [topic for article in articles for topic in article.get("topics", [])]
        topic_counter = Counter(all_topics)
        file_path = plot_topic_frequency(topic_counter)
        return FileResponse(file_path)
    else:
        return JSONResponse(status_code=404, content={"message": f"Visualization type '{viz_type}' not found"})
