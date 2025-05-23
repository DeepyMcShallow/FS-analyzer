# sentiment_analyzer.py
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is downloaded (only needs to be done once)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    print("Downloading VADER lexicon for NLTK (one-time download)...")
    nltk.download('vader_lexicon')
    print("Download complete.")

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    """
    Analyzes the sentiment of a given text using VADER.
    Returns a dictionary with label, compound score, and individual scores.
    """
    if not text or not isinstance(text, str):
        return {"label": "Neutral", "compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}

    vs = analyzer.polarity_scores(text)
    compound_score = vs["compound"]
    
    label = "Neutral"
    if compound_score >= 0.05: # Standard VADER threshold
        label = "Positive"
    elif compound_score <= -0.05:
        label = "Negative"
    
    return {
        "label": label,
        "compound": compound_score,
        "pos": vs["pos"],
        "neu": vs["neu"],
        "neg": vs["neg"]
    }

def classify_sentiment_strength(compound_score):
    """Classifies sentiment strength based on compound score."""
    if compound_score >= 0.7: return "Very Positive"
    if 0.3 <= compound_score < 0.7: return "Moderate Positive"
    if -0.3 < compound_score < 0.3: return "Neutral"
    if -0.7 < compound_score <= -0.3: return "Moderate Negative"
    if compound_score <= -0.7: return "Very Negative"
    return "Neutral" # Default

if __name__ == '__main__':
    print("--- Testing Sentiment Analyzer ---")
    test_texts = [
        "This is great news, profits are soaring!",
        "The company reported a significant loss and stock prices plummeted.",
        "The report was largely as expected with no major surprises.",
        "Uncertainty looms over the market due to new regulations.",
        "The product launch was a spectacular success, exceeding all expectations."
    ]
    for text in test_texts:
        sentiment_result = get_sentiment(text)
        strength = classify_sentiment_strength(sentiment_result['compound'])
        print(f"\nText: {text}")
        print(f"  Sentiment: {sentiment_result['label']} (Compound: {sentiment_result['compound']:.2f})")
        print(f"  Strength: {strength}")
        print(f"  Scores: Pos={sentiment_result['pos']:.2f}, Neu={sentiment_result['neu']:.2f}, Neg={sentiment_result['neg']:.2f}")