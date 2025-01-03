from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
import numpy as np
import joblib
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import feedparser
import requests
import json
import subprocess
import threading
import time
from difflib import SequenceMatcher
import tempfile
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import assemblyai as aai
from dotenv import load_dotenv
import queue
import concurrent.futures
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='public')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Constants
PORT = int(os.getenv('PORT', 3000))
NEWS_UPDATE_INTERVAL = 300  # 5 minutes
CACHE_CLEANUP_INTERVAL = 3600  # 1 hour

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Initialize AssemblyAI
aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')

# Dummy dataset for training if needed
DUMMY_TEXTS = [
    "This is a sample news article",
    "Another example of fake news",
    "Real news article about science"
]
DUMMY_LABELS = [0, 1, 0]

# Global pipeline object; we'll assign it when the app starts
pipeline = None

def train_and_save_pipeline():
    """Train a simple pipeline and save it to disk."""
    vectorizer = TfidfVectorizer()
    model = LogisticRegression()
    pipe = Pipeline([('tfidf', vectorizer), ('classifier', model)])
    pipe.fit(DUMMY_TEXTS, DUMMY_LABELS)
    joblib.dump(pipe, 'saved_models/pipeline.pkl')
    logger.info("Pipeline trained and saved.")

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """
    Analyzes input text for a 'fake news' classification (1 = fake, 0 = real).
    Returns JSON with prediction and confidence score.
    """
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400

        text = data['text']
        processed_text = pipeline.transform([text])
        prediction = pipeline.predict(processed_text)
        confidence = pipeline.predict_proba(processed_text).max()

        return jsonify({
            'text': text,
            'analysis': {
                'prediction': bool(prediction[0]),
                'confidence': float(confidence)
            },
            'success': True
        })
    except Exception as e:
        logger.error(f'Text analysis error: {str(e)}')
        return jsonify({'error': 'Failed to analyze text', 'details': str(e)}), 500

if __name__ == '__main__':
    try:
        # Create necessary directories if they don't exist
        Path('saved_models').mkdir(exist_ok=True)
        Path('public').mkdir(exist_ok=True)

        # Ensure we have an index.html for the static folder
        index_path = Path('public/index.html')
        if not index_path.exists():
            with open(index_path, 'w') as f:
                f.write("""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>News Analysis Service</title>
                </head>
                <body>
                    <h1>News Analysis Service</h1>
                    <p>API is running successfully.</p>
                </body>
                </html>
                """)

        # Load or train pipeline inside main to ensure directories exist
        pipeline_path = 'saved_models/pipeline.pkl'
        if not os.path.exists(pipeline_path):
            logger.warning("Pipeline not found. Training a new one...")
            train_and_save_pipeline()
        pipeline = joblib.load(pipeline_path)

        logger.info(f'Server running at http://localhost:{PORT}')
        socketio.run(app, host='0.0.0.0', port=PORT, debug=False)

    except Exception as e:
        logger.error(f'Server startup error: {str(e)}')
        raise
