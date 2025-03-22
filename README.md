# News Sentiment Analyzer

## Project Setup

### Prerequisites

Ensure you have Python 3.8 or later installed.

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/Anirudh973/Akaike_Internship.git
   cd Akaike_Internship
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your NewsAPI key:

   ```env
   NEWS_API_KEY=your_api_key_here
   ```

### Running the Application

#### Start the FastAPI Backend

```bash
uvicorn api:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

#### Start the Streamlit Frontend

```bash
streamlit run app.py
```

The application UI will open in your default browser.

## Model Details

### Summarization Model

- Uses NLP techniques to summarize news articles based on sentiment distribution.
- Summary is generated based on topic frequency and sentiment dominance.

### Sentiment Analysis Model

- Utilizes `transformers` for text classification.
- Identifies sentiments as Positive, Negative, or Neutral.

### Text-to-Speech (TTS) Model

- Uses `gTTS` (Google Text-to-Speech) to generate Hindi audio summaries.

## API Development

- Built using FastAPI.
- Endpoints are structured to fetch, analyze, and visualize news data.
- Can be tested using Postman or directly via browser requests.

### API Endpoints

- `GET /news/{company}` - Fetches news articles and performs sentiment analysis.
- `GET /tts/{company}` - Generates a Hindi TTS summary.
- `GET /news/{company}/report` - Provides a detailed analysis report.
- `POST /news/{company}/query` - Queries news articles based on user input.
- `GET /news/{company}/visualization/{viz_type}` - Returns visualizations like sentiment distribution, word cloud, or topic frequency.

## API Usage

### Third-Party API Integration

- **NewsAPI**: Used to fetch news articles based on a company name.
  - Endpoint: `https://newsapi.org/v2/everything`
  - Requires an API key stored in the `.env` file.

## Assumptions & Limitations

### Assumptions

- News articles fetched are in English.
- Sentiment analysis is based on headlines and descriptions.
- The API key provided has sufficient request limits for testing.

### Limitations

- Sentiment analysis may not always be 100% accurate.
- NewsAPI has rate limits, which may restrict frequent requests.
- TTS generation is limited to Hindi language only.

## Technologies Used

- **Backend:** FastAPI, Uvicorn, Requests
- **Frontend:** Streamlit
- **Machine Learning:** Transformers, KeyBERT, Scikit-learn
- **NLP & Visualization:** Matplotlib, Seaborn, WordCloud, gTTS
- **Data Handling:** Pandas, NumPy

## Author

Developed by Anirudh Rajagopal for Akaike

