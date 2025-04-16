import streamlit as st
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime
from app import db, app
import logging
from langdetect import detect, detect_langs
from deepseek_analyzer import DeepSeekAnalyzer

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('vader_lexicon')

# Load NLP models
try:
    nlp = spacy.load('en_core_web_sm')
except:
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Initialize sentiment analyzer and DeepSeek
sia = SentimentIntensityAnalyzer()
deepseek = DeepSeekAnalyzer()

def get_website_content(url):
    """Extract content from URL using BeautifulSoup"""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        logger.error(f"Error extracting content from URL: {str(e)}")
        return None

def detect_language(text):
    """Detect the language of the text using langdetect"""
    try:
        # Get detailed language probabilities
        langs = detect_langs(text)
        primary_lang = langs[0]

        # Map language codes to full names
        language_names = {
            'hi': 'Hindi',
            'mr': 'Marathi',
            'en': 'English',
            'ta': 'Tamil',
            'te': 'Telugu',
            'bn': 'Bengali',
            'gu': 'Gujarati',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'pa': 'Punjabi',
            'ur': 'Urdu'
        }

        lang_name = language_names.get(primary_lang.lang, primary_lang.lang)
        return {
            'language': lang_name,
            'code': primary_lang.lang,
            'score': primary_lang.prob
        }
    except Exception as e:
        logger.error(f"Error detecting language: {str(e)}")
        return {
            'language': 'Unknown',
            'code': 'unknown',
            'score': 0.0
        }

def analyze_bias(text):
    """Analyze text for potential bias"""
    # Bias indicators (simplified version)
    bias_indicators = {
        'emotional_language': ['absolutely', 'clearly', 'obviously', 'undoubtedly'],
        'loaded_words': ['radical', 'extremist', 'fanatic', 'terrorist'],
        'generalizations': ['all', 'every', 'always', 'never', 'none'],
        'subjective_phrases': ['should', 'must', 'need to', 'have to']
    }

    bias_score = 0
    bias_details = {}

    # Count occurrences of bias indicators
    for category, words in bias_indicators.items():
        matches = sum(1 for word in words if word.lower() in text.lower())
        if matches > 0:
            bias_score += matches * 5  # Each bias indicator adds 5 points
            bias_details[category] = matches

    return {
        'score': min(bias_score, 100),  # Cap at 100
        'details': bias_details
    }

def check_universal_truths(text):
    """Check if text contains universal scientific truths"""
    universal_truths = {
        "Earth is round": 100,
        "Gravity exists": 100,
        "Water is H2O": 100,
        "Sun rises in the east": 100,
        "Earth revolves around the sun": 100,
        "Earth has one moon": 100
    }

    # Check if text matches any universal truth
    for truth, score in universal_truths.items():
        if truth.lower() in text.lower():
            return score
    return None

def analyze_credibility(text):
    """Analyze text credibility using multiple NLP techniques"""
    # Check for universal truths first
    truth_score = check_universal_truths(text)
    if truth_score is not None:
        return {
            'score': truth_score,
            'category': 'Highly Credible',
            'sentiment': sia.polarity_scores(text),
            'named_entities': 0,
            'language': detect_language(text),
            'deepseek_analysis': {'api_response': False},
            'bias_analysis': {'score': 0, 'details': {'universal_truth': True}},
            'text_stats': {
                'word_count': len(text.split()),
                'entity_density': 0
            }
        }

    # Load URL databases
    fake_news_db = pd.read_csv('attached_assets/fake_news_websites.csv')
    credible_news_db = pd.read_csv('attached_assets/credible_news_websites.csv')

    # Check if content is URL or contains URLs
    urls = []
    if text.startswith(('http://', 'https://')):
        urls.append(text)
    else:
        # Extract URLs from text content
        import re
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)

    # Check URLs against databases
    for url in urls:
        # Check against fake news database
        if any(url.startswith(fake_url) for fake_url in fake_news_db['URL']):
            logger.info(f"URL {url} found in fake news database")
            return {
                'score': float(fake_news_db[fake_news_db['URL'].apply(lambda x: url.startswith(x))]['Trust Score'].iloc[0]),
                'category': 'Potentially Fake',
                'sentiment': sia.polarity_scores(text),
                'named_entities': 0,
                'language': detect_language(text),
                'deepseek_analysis': {'api_response': False},
                'bias_analysis': {'score': 100, 'details': {'found_in_fake_db': True}},
                'text_stats': {
                    'word_count': len(text.split()),
                    'entity_density': 0
                }
            }

        # Check against credible news database
        if any(url.startswith(cred_url) for cred_url in credible_news_db['URL']):
            logger.info(f"URL {url} found in credible news database")
            return {
                'score': float(credible_news_db[credible_news_db['URL'].apply(lambda x: url.startswith(x))]['Trust Score'].iloc[0]),
                'category': 'Highly Credible',
                'sentiment': sia.polarity_scores(text),
                'named_entities': 0,
                'language': detect_language(text),
                'deepseek_analysis': {'api_response': False},
                'bias_analysis': {'score': 0, 'details': {'found_in_credible_db': True}},
                'text_stats': {
                    'word_count': len(text.split()),
                    'entity_density': 0
                }
            }

    # If no URLs found or URLs not in databases, proceed with normal analysis
    language = detect_language(text)
    logger.info(f"Detected language: {language}")

    # Get DeepSeek analysis
    deepseek_analysis = deepseek.analyze_content(text)

    # Sentiment analysis with VADER (works best with English)
    sentiment_scores = sia.polarity_scores(text)

    # Named Entity Recognition with spaCy
    doc = nlp(text)
    named_entities = len(doc.ents)

    # Analyze bias
    bias_analysis = analyze_bias(text)

    # Calculate credibility score based on multiple factors
    base_score = 50

    # Factor in sentiment (extreme sentiment reduces score)
    sentiment_impact = abs(sentiment_scores['compound']) * 20
    base_score -= sentiment_impact

    # Factor in named entities (more entities usually indicate more detailed content)
    base_score += min(named_entities, 10) * 2

    # Factor in text length (longer texts with entities are usually more detailed)
    text_length_score = min(len(text.split()) / 100, 10)
    base_score += text_length_score

    # Factor in bias (high bias reduces score)
    bias_impact = bias_analysis['score'] * 0.2  # 20% weight for bias
    base_score -= bias_impact

    # Add DeepSeek analysis impact if available
    if deepseek_analysis['api_response']:
        base_score += 10  # Bonus for successful deep analysis

    # Ensure score is between 0 and 100
    final_score = min(max(base_score, 0), 100)

    return {
        'score': round(final_score, 1),
        'category': get_credibility_category(final_score),
        'sentiment': sentiment_scores,
        'named_entities': named_entities,
        'language': language,
        'deepseek_analysis': deepseek_analysis,
        'bias_analysis': bias_analysis,
        'text_stats': {
            'word_count': len(text.split()),
            'entity_density': named_entities / max(len(text.split()) / 100, 1)
        }
    }

def get_credibility_category(score):
    """Get credibility category based on score"""
    if score >= 80:
        return "Highly Credible"
    elif score >= 50:
        return "Partially Verified"
    else:
        return "Potentially Fake"

# Set page config with custom theme
st.set_page_config(
    page_title="Sach.AI - Multilingual Fact Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for animations and styling
st.markdown("""
<style>
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes scoreAnimation {
    from { transform: scale(0.8); opacity: 0; }
    to { transform: scale(1); opacity: 1; }
}
.fade-in {
    animation: fadeIn 1s ease-in;
}
.score-animate {
    animation: scoreAnimation 1.5s ease-out;
}
.tooltip {
    position: relative;
    display: inline-block;
}
.tooltip .tooltiptext {
    visibility: hidden;
    background-color: rgba(0, 0, 0, 0.8);
    color: #fff;
    text-align: center;
    padding: 8px 12px;
    border-radius: 6px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    opacity: 0;
    transition: opacity 0.3s;
    white-space: nowrap;
    font-size: 0.9em;
}
.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
</style>
""", unsafe_allow_html=True)

# Language selection
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# Welcome message based on language
welcome_messages = {
    'en': "👋 Welcome to Sach.AI! I'm here to help you verify news and information.",
    'hi': "👋 Sach.AI में आपका स्वागत है! मैं समाचारों की सत्यता की जांच में आपकी मदद करूंगा।",
    'mr': "👋 Sach.AI वर आपले स्वागत आहे! मी बातम्यांची सत्यता तपासण्यात आपली मदत करेन।"
}

tooltips = {
    'en': {
        'url': "Enter a news article URL to analyze its content",
        'text': "Or paste the text content directly for analysis",
        'score': "Our AI-powered credibility assessment",
        'bias': "Analysis of potential bias in the content",
        'verify': "Click to verify from multiple sources"
    },
    'hi': {
        'url': "सामग्री का विश्लेषण करने के लिए समाचार लेख URL दर्ज करें",
        'text': "या विश्लेषण के लिए सीधे पाठ सामग्री पेस्ट करें",
        'score': "हमारा AI-संचालित विश्वसनीयता मूल्यांकन",
        'bias': "सामग्री में संभावित पूर्वाग्रह का विश्लेषण",
        'verify': "कई स्रोतों से सत्यापित करने के लिए क्लिक करें"
    },
    'mr': {
        'url': "सामग्री विश्लेषण करण्यासाठी बातमी लेख URL टाका",
        'text': "किंवा विश्लेषणासाठी थेट मजकूर पेस्ट करा",
        'score': "आमचे AI-आधारित विश्वसनीयता मूल्यांकन",
        'bias': "सामग्रीमधील संभाव्य पूर्वग्रहांचे विश्लेषण",
        'verify': "अनेक स्रोतांमधून सत्यापित करण्यासाठी क्लिक करा"
    }
}

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Add chat control buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("New Chat"):
        st.session_state.messages = []
        st.rerun()
with col2:
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

# Language selector in sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/feathericons/feather/master/icons/check-circle.svg", width=50)
    selected_lang = st.selectbox(
        "Choose Language / भाषा चुनें / भाषा निवडा",
        ['en', 'hi', 'mr'],
        format_func=lambda x: {'en': 'English', 'hi': 'हिंदी', 'mr': 'मराठी'}[x]
    )
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()

# Display welcome message if no messages
if not st.session_state.messages:
    st.markdown(f'<div class="fade-in">{welcome_messages[st.session_state.language]}<br><br>|| सत्यमेव जयते || Truth Alone Triumphs ||</div>', unsafe_allow_html=True)

st.title("Sach.AI - Secure Artificial Intelligence for Checking Hoaxes")

# Add option for web search
search_mode = st.radio(
    "Select Analysis Mode",
    ["Direct Analysis", "Web Search Analysis"]
)

if search_mode == "Web Search Analysis":
    st.info("This will analyze top 10 Google search results for the given query")
    # Import at the top level to catch any import errors early
    import search_api

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
input_placeholder = {
    'en': "Enter news text or URL to analyze...",
    'hi': "विश्लेषण के लिए समाचार टेक्स्ट या URL दर्ज करें...",
    'mr': "विश्लेषणासाठी बातमी मजकूर किंवा URL टाका..."
}

if prompt := st.chat_input(input_placeholder[st.session_state.language]):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Determine if input is URL
    is_url = prompt.startswith(('http://', 'https://'))

    # Get content
    try:
        # Handle web search mode if selected
        if search_mode == "Web Search Analysis":
            # Show analysis in progress
            with st.chat_message("assistant"):
                with st.spinner("Searching and analyzing content..."):
                    try:
                        # Perform search analysis
                        search_result = search_api.analyze_search_results(prompt)
                        
                        # Convert search result to standard format
                        result = {
                            'score': search_result['score'],
                            'category': search_result['category'],
                            'sentiment': {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0},
                            'named_entities': 0,
                            'language': {'language': 'English', 'code': 'en', 'score': 1.0},
                            'deepseek_analysis': {'api_response': False},
                            'bias_analysis': {'score': 0, 'details': {}},
                            'text_stats': {'word_count': len(prompt.split()), 'entity_density': 0},
                            'search_results': search_result.get('search_results', [])
                        }
                        
                        # Log successful search
                        logger.info(f"Web search completed successfully for: {prompt}")
                    except Exception as search_error:
                        logger.error(f"Error in web search: {str(search_error)}", exc_info=True)
                        st.error(f"Error in web search: {str(search_error)}")
                        # Fall back to direct analysis
                        result = analyze_credibility(prompt)
        else:
            # Regular direct analysis
            if is_url:
                content = get_website_content(prompt)
                if not content:
                    raise ValueError("Could not extract content from URL")
            else:
                content = prompt

            # Show analysis in progress
            with st.chat_message("assistant"):
                with st.spinner("Analyzing content..."):
                    # Analyze content
                    result = analyze_credibility(content)

                # Store in database using Flask app context - only if database is available
                try:
                    with app.app_context():
                        if hasattr(app, 'db_available') and app.db_available:
                            from models import NewsArticle, FactCheckResult
                            article = NewsArticle(
                                url=prompt if is_url else None,
                                content=content[:500]  # Store first 500 chars
                            )
                            db.session.add(article)
                            db.session.flush()

                            fact_check = FactCheckResult(
                                article_id=article.id,
                                credibility_score=result['score'],
                                category=result['category'],
                                verified_claims=0,
                                questionable_claims=0,
                                sentiment_scores=result['sentiment']
                            )
                            db.session.add(fact_check)
                            db.session.commit()
                except Exception as db_error:
                    logger.info(f"Skipping database operations: {str(db_error)}")
                    # Continue without database operations

                # Display results with animations
                st.markdown(f"""
                <div class="score-animate">
                    <h3>
                        <div class="tooltip">
                            Credibility Score: {result['score']}%
                            <span class="tooltiptext">{tooltips[st.session_state.language]['score']}</span>
                        </div>
                    </h3>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="fade-in">
                    <div class="tooltip">
                        <strong>Category:</strong> {result['category']}
                        <span class="tooltiptext">Based on our comprehensive analysis</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="fade-in">
                    <div class="tooltip">
                        <strong>Detected Language:</strong> {result['language']['language']} 
                        (Confidence: {result['language']['score']:.2%})
                        <span class="tooltiptext">Automatically detected content language</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Display bias analysis
                st.markdown(f"""
                <div class="fade-in">
                    <div class="tooltip">
                        <strong>Bias Score:</strong> {result['bias_analysis']['score']}%
                        <span class="tooltiptext">{tooltips[st.session_state.language]['bias']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Display DeepSeek analysis if available
                if result['deepseek_analysis']['api_response']:
                    st.markdown("### Deep Analysis")
                    st.markdown(result['deepseek_analysis']['topic_analysis'])

                # Display search results if available
                if search_mode == "Web Search Analysis" and 'search_results' in result:
                    st.subheader("Search Results")
                    for idx, res in enumerate(result.get('search_results', [])):
                        with st.expander(f"{idx+1}. {res.get('title', 'Result')}"):
                            st.write(f"**Source:** {res.get('source', 'Unknown')}")
                            st.write(f"**Link:** {res.get('link', '#')}")
                            st.write(f"**Summary:** {res.get('snippet', 'No summary available')}")
                
                # Display detailed analysis
                with st.expander("View Detailed Analysis"):
                    analysis_data = {
                        'sentiment_analysis': result['sentiment'],
                        'named_entities_found': result['named_entities'],
                        'language_detection': result['language'],
                        'bias_analysis': result['bias_analysis'],
                        'text_statistics': result['text_stats']
                    }
                    if search_mode == "Web Search Analysis":
                        analysis_data['search_mode'] = "Web Search"
                    st.json(analysis_data)

                # Add response to chat history
                response = f"""
                🙏 Namaste!

                📊 **Credibility Analysis Results**
                - Score: {result['score']}%
                - Category: {result['category']}
                - Language: {result['language']['language']}
                - Bias Score: {result['bias_analysis']['score']}%
                - Sentiment: {'Positive' if result['sentiment']['compound'] > 0 else 'Negative'}
                - Named Entities: {result['named_entities']}

                || सत्यमेव जयते || Truth Alone Triumphs ||
                """
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Add feedback section
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Helpful"):
                        st.success("Thank you for your feedback!")
                with col2:
                    if st.button("👎 Not Helpful"):
                        feedback = st.text_area("Please tell us what went wrong:")
                        if st.button("Submit Feedback"):
                            st.success("Thank you for your feedback! We'll work on improving.")

    except Exception as e:
        error_message = f"Error analyzing content: {str(e)}"
        st.error(error_message)
        logger.error(f"Error in content analysis: {str(e)}", exc_info=True)
        st.session_state.messages.append({"role": "assistant", "content": f"❌ {error_message}"})

# Add sidebar with information
with st.sidebar:
    st.markdown("""
    ### About Sach.AI
    A fact-checking and credibility analysis tool.
    """)