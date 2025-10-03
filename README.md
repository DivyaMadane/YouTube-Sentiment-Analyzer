# YouTube Sentiment Analyzer

A powerful web application that analyzes the sentiment of YouTube video comments using advanced Natural Language Processing (NLP) techniques. The application fetches comments from YouTube videos and provides detailed sentiment analysis with beautiful visualizations.

## 🌟 Features

- **YouTube Comment Fetching**: Automatically extracts and fetches comments from any YouTube video URL
- **Multilingual Support**: Detects comment language and translates non-English comments to English for accurate analysis
- **Dual Sentiment Analysis**: Supports both VADER and TextBlob algorithms for sentiment analysis
- **Interactive Visualizations**: 
  - Sentiment distribution pie charts and bar charts
  - Word clouds for positive and negative comments
  - Detailed comment analysis table
- **Export Functionality**: Download complete analysis results as CSV files
- **Modern Web Interface**: Beautiful, responsive design with real-time analysis
- **Emoji-Aware Processing**: Converts emojis to sentiment hints for better accuracy

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- YouTube Data API v3 key (free from Google Cloud Console)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DivyaMadane/YouTube-Sentiment-Analyzer.git
   cd youtube_sentiment_analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up YouTube API Key**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one
   - Enable YouTube Data API v3
   - Create credentials (API Key)
   - Replace the API key in `backend/fetch_comments.py`:
     ```python
     API_KEY: str = "YOUR_API_KEY_HERE"
     ```

4. **Run the application**
   ```bash
   cd frontend
   python app.py
   ```

5. **Access the application**
   - Open your browser and go to `http://127.0.0.1:5000`
   - Enter a YouTube video URL and start analyzing!

## 📁 Project Structure

```
youtube_sentiment_analyzer/
├── backend/
│   ├── fetch_comments.py      # YouTube API integration and comment fetching
│   └── sentiment_analysis.py  # NLP processing and sentiment analysis
├── frontend/
│   ├── app.py                 # Flask web application
│   └── templates/
│       └── index.html         # Web interface
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🔧 How It Works

### 1. Comment Fetching (`backend/fetch_comments.py`)
- Extracts video ID from various YouTube URL formats
- Uses YouTube Data API v3 to fetch comments
- Supports multiple URL formats:
  - `https://www.youtube.com/watch?v=VIDEOID`
  - `https://youtu.be/VIDEOID`
  - `https://www.youtube.com/shorts/VIDEOID`
  - And more...

### 2. Sentiment Analysis (`backend/sentiment_analysis.py`)
- **Language Detection**: Uses `langdetect` to identify comment language
- **Translation**: Translates non-English comments using Google Translate
- **Text Preprocessing**: 
  - Removes URLs and handles
  - Converts emojis to sentiment hints
  - Cleans and normalizes text
- **Sentiment Analysis**: Supports two algorithms:
  - **VADER**: Lexicon-based, great for social media text
  - **TextBlob**: Machine learning-based, good for formal text

### 3. Web Interface (`frontend/app.py`)
- Flask-based web application
- Real-time analysis with AJAX
- Generates visualizations using matplotlib and seaborn
- Creates word clouds for positive/negative comments
- Exports results as CSV

## 📊 Sentiment Analysis Methods

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Type**: Lexicon-based approach
- **Strengths**: Excellent for social media, handles emojis, punctuation, and capitalization
- **Score Range**: -1.0 (most negative) to +1.0 (most positive)
- **Example**: "This video is absolutely amazing!!!" → Score: 0.83 (Positive)

### TextBlob
- **Type**: Machine learning approach
- **Strengths**: Good for formal text, trained on movie reviews
- **Score Range**: -1.0 (most negative) to +1.0 (most positive)
- **Example**: "Great tutorial, very helpful" → Score: 0.65 (Positive)

### Classification Thresholds
- **Positive**: Score ≥ 0.05
- **Neutral**: -0.05 < Score < 0.05
- **Negative**: Score ≤ -0.05

## 🌍 Multilingual Support

The application automatically handles comments in multiple languages:

1. **Language Detection**: Identifies the language of each comment
2. **Translation**: Translates non-English comments to English
3. **Analysis**: Performs sentiment analysis on English text
4. **Preprocessing**: Handles emojis, URLs, and special characters

## 📈 Visualizations

- **Sentiment Distribution**: Pie chart showing percentage breakdown
- **Sentiment Counts**: Bar chart with absolute numbers
- **Word Clouds**: Visual representation of most common words in positive/negative comments
- **Detailed Table**: Complete analysis with scores, translations, and metadata

## 🔧 Configuration

### API Limits
- YouTube Data API v3 has daily quotas
- Default comment limit: 100 (configurable 25-500)
- For higher limits, consider YouTube API premium

### Performance Optimization
- Reduced chart sizes for faster generation
- Limited word cloud generation (minimum 20 words)
- Optimized matplotlib settings for web display

## 📋 Dependencies

```
pandas>=2.0.0
numpy>=1.23.0
google-api-python-client>=2.129.0
textblob>=0.17.1
vaderSentiment>=3.3.2
nltk>=3.8.1
Flask>=3.0.0
matplotlib>=3.7.0
seaborn>=0.12.2
wordcloud>=1.9.2
googletrans==4.0.0-rc1
langdetect>=1.0.9
```

## 🚨 Important Notes

### API Key Security
- **Never commit your API key to version control**
- Consider using environment variables for production
- Monitor your API usage in Google Cloud Console

### Rate Limits
- YouTube API has daily quotas (10,000 units/day for free tier)
- Each comment fetch costs 1 unit
- Monitor usage to avoid hitting limits

### Data Privacy
- Comments are fetched from public YouTube videos only
- No personal data is stored permanently
- Analysis is performed in real-time

## 🛠️ Troubleshooting

### Common Issues

1. **"Could not extract video ID"**
   - Ensure the URL is a valid YouTube video URL
   - Check if the video is public and has comments enabled

2. **"YouTube API error"**
   - Verify your API key is correct
   - Check if YouTube Data API v3 is enabled
   - Ensure you haven't exceeded daily quotas

3. **"No comments found"**
   - Some videos may have comments disabled
   - Try a different video with active comments

4. **Translation errors**
   - Some comments may fail to translate
   - The system falls back to original text in such cases

## 🔮 Future Enhancements

- [ ] Support for YouTube Shorts and Live streams
- [ ] Real-time comment monitoring
- [ ] Advanced sentiment analysis models (BERT, RoBERTa)
- [ ] Comment thread analysis (replies)
- [ ] Historical sentiment tracking
- [ ] Batch video analysis
- [ ] API endpoint for external integrations

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📞 Support

If you encounter any issues or have questions, please:
1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed information

---

**Happy Analyzing! 🎉**

*Built with ❤️ using Python, Flask, and modern NLP techniques*
