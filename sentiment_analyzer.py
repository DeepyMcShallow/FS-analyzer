# sentiment_analyzer.py
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is downloaded 
# This needs to run when the app starts up on Streamlit Cloud if the lexicon isn't there.
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
    # print("VADER lexicon found.") # Optional: for debugging successful find
except LookupError: # This is the more standard exception if resource is not found
    print("VADER lexicon not found. Attempting to download for NLTK...")
    try:
        nltk.download('vader_lexicon')
        print("VADER lexicon download complete.")
        # Verify it's there now
        nltk.data.find('sentiment/vader_lexicon.zip')
        print("VADER lexicon successfully verified after download.")
    except Exception as e:
        # This could be due to network issues on the server or write permission issues (less likely on Streamlit Cloud for nltk_data)
        print(f"ERROR: Failed to download or verify VADER lexicon after attempting download: {e}")
        print("Sentiment analysis may not work correctly without the VADER lexicon.")
        # Depending on how critical this is, you might raise an error or allow the app to continue with a warning.
        # For now, we'll let it try to initialize Analyzer, which might fail if lexicon is truly unavailable.
except Exception as e:
    print(f"An unexpected error occurred while checking for VADER lexicon: {e}")


# Initialize the analyzer *after* attempting the download
try:
    analyzer = SentimentIntensityAnalyzer()
except LookupError: # Catch if VADER still can't be initialized (e.g., download failed silently or other issue)
    print("CRITICAL ERROR: SentimentIntensityAnalyzer could not be initialized. VADER lexicon might still be missing or corrupted.")
    print("Please check app logs on Streamlit Cloud. Sentiment analysis will not function.")
    # Define a dummy analyzer or raise an exception to stop the app if sentiment is critical
    class DummySentimentIntensityAnalyzer:
        def polarity_scores(self, text):
            print("Warning: Using DUMMY sentiment analyzer due to initialization failure.")
            return {"label": "Neutral (Error)", "compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}
    analyzer = DummySentimentIntensityAnalyzer()


def get_sentiment(text):
    """
    Analyzes the sentiment of a given text using VADER.
    Returns a dictionary with label, compound score, and individual scores.
    """
    if not text or not isinstance(text, str):
        return {"label": "Neutral", "compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}

    vs = analyzer.polarity_scores(text) # Uses the globally defined 'analyzer'
    compound_score = vs["compound"]
    
    label = "Neutral"
    # Using the VADER standard thresholds for positive/negative
    if compound_score >= 0.05: 
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
    # These are the thresholds we fine-tuned previously
    if compound_score >= 0.7: return "Very Positive"
    if 0.05 <= compound_score < 0.7: return "Moderate Positive" 
    if -0.05 < compound_score < 0.05: return "Neutral" 
    if -0.7 < compound_score <= -0.05: return "Moderate Negative"
    if compound_score <= -0.7: return "Very Negative"
    return "Neutral"

if __name__ == '__main__':
    print("--- Testing Sentiment Analyzer ---")
    # Test the download logic (if run locally and lexicon isn't there)
    # analyzer should be initialized by this point if successful

    test_texts = [
        "This is great news, profits are soaring!",
        "The company reported a significant loss and stock prices plummeted.",
        "The report was largely as expected with no major surprises.",
        "Despite some concerns, the overall outlook remains quite bright and promising.",
        "Market sentiment is rather gloomy today after the announcement."
    ]
    for text in test_texts:
        sentiment_result = get_sentiment(text)
        strength = classify_sentiment_strength(sentiment_result['compound'])
        print(f"\nText: {text}")
        print(f"  Sentiment: {sentiment_result['label']} (Compound: {sentiment_result['compound']:.2f})")
        print(f"  Strength: {strength}")
        print(f"  Scores: Pos={sentiment_result['pos']:.2f}, Neu={sentiment_result['neu']:.2f}, Neg={sentiment_result['neg']:.2f}")
