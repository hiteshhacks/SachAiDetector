from datetime import datetime
from app import db
from sqlalchemy import Enum

class NewsArticle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(2048), nullable=True)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    fact_check_results = db.relationship('FactCheckResult', back_populates='article')

class FactCheckResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    article_id = db.Column(db.Integer, db.ForeignKey('news_article.id'), nullable=False)
    credibility_score = db.Column(db.Float, nullable=False)
    category = db.Column(
        Enum('Highly Credible', 'Partially Verified', 'Potentially Fake', name='credibility_category'),
        nullable=False
    )
    verified_claims = db.Column(db.Integer)
    questionable_claims = db.Column(db.Integer)
    sentiment_scores = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    article = db.relationship('NewsArticle', back_populates='fact_check_results')
