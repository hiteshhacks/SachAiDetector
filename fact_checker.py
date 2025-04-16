import requests
import trafilatura
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import logging

# Download required NLTK data
nltk.download('vader_lexicon')

logger = logging.getLogger(__name__)

def get_website_content(url):
    """Extract content from URL using trafilatura"""
    try:
        downloaded = trafilatura.fetch_url(url)
        return trafilatura.extract(downloaded)
    except Exception as e:
        logger.error(f"Error extracting content from URL: {str(e)}")
        return None

def check_facts_with_google_api(text):
    """Check facts using Google Fact Check API"""
    # Note: In a real implementation, you would use the actual Google Fact Check API
    # This is a simplified version for demonstration
    return {
        'claims_found': 2,
        'verified_claims': 1,
        'questionable_claims': 1
    }

def analyze_sentiment(text):
    """Analyze sentiment of the text"""
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores

def calculate_credibility_score(fact_check_results, sentiment_scores):
    """Calculate overall credibility score"""
    # Basic scoring algorithm
    base_score = 50
    
    # Factor in fact checking results
    if fact_check_results['claims_found'] > 0:
        verification_ratio = fact_check_results['verified_claims'] / fact_check_results['claims_found']
        base_score += (verification_ratio * 30)
    
    # Factor in sentiment analysis
    if abs(sentiment_scores['compound']) > 0.5:
        # Highly polarized content reduces score
        base_score -= 10
    
    return min(max(base_score, 0), 100)  # Ensure score is between 0 and 100

def analyze_content(content, content_type='text'):
    """Main analysis function"""
    try:
        # Get content from URL if needed
        if content_type == 'url':
            text_content = get_website_content(content)
            if not text_content:
                raise ValueError("Could not extract content from URL")
        else:
            text_content = content

        # Perform analysis
        fact_check_results = check_facts_with_google_api(text_content)
        sentiment_scores = analyze_sentiment(text_content)
        credibility_score = calculate_credibility_score(fact_check_results, sentiment_scores)

        # Determine credibility category
        if credibility_score >= 80:
            category = "Highly Credible"
            category_class = "success"
        elif credibility_score >= 50:
            category = "Partially Verified"
            category_class = "warning"
        else:
            category = "Potentially Fake"
            category_class = "danger"

        return {
            'score': round(credibility_score, 1),
            'category': category,
            'category_class': category_class,
            'fact_check_results': fact_check_results,
            'sentiment': sentiment_scores,
            'analyzed_text': text_content[:500] + '...' if len(text_content) > 500 else text_content
        }

    except Exception as e:
        logger.error(f"Error in content analysis: {str(e)}")
        raise
