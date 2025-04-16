
import os
import logging
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from fact_checker import analyze_content

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize SQLAlchemy with a custom base
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

# Configure database - Set to SQLite if no DB URL provided
db_url = os.environ.get("DATABASE_URL")
if not db_url:
    db_url = "sqlite:///fact_checker.db"

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize database
db.init_app(app)

# Flag to check if database is available
db_available = True

# Import models after db initialization
try:
    with app.app_context():
        from models import NewsArticle, FactCheckResult
        db.create_all()
except Exception as e:
    logger.error(f"Database initialization error: {str(e)}")
    db_available = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        content = request.form.get('content', '')
        content_type = request.form.get('type', 'text')  # 'text' or 'url'

        if not content:
            return jsonify({'error': 'No content provided'}), 400

        # Analyze the content
        result = analyze_content(content, content_type)

        # Store the analysis in the database only if available
        if db_available:
            try:
                with app.app_context():
                    # Create news article entry
                    article = NewsArticle(
                        url=content if content_type == 'url' else None,
                        content=result['analyzed_text']
                    )
                    db.session.add(article)
                    db.session.flush()  # Get the article ID

                    # Create fact check result entry
                    fact_check = FactCheckResult(
                        article_id=article.id,
                        credibility_score=result['score'],
                        category=result['category'],
                        verified_claims=result['fact_check_results']['verified_claims'],
                        questionable_claims=result['fact_check_results']['questionable_claims'],
                        sentiment_scores=result['sentiment']
                    )
                    db.session.add(fact_check)
                    db.session.commit()
            except Exception as db_error:
                logger.error(f"Database storage error: {str(db_error)}")
                # Continue without database storage

        return render_template('result.html', result=result)

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        return jsonify({'error': 'An error occurred during analysis'}), 500
