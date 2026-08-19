import os
import json
import joblib
import numpy as np
from train_model import clean_text

class CyberbullyingPredictor:
    def __init__ (self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.base_dir, 'model', 'cyberbullying_model.pkl')
        self.vectorizer_path = os.path.join(self.base_dir, 'model', 'tfidf_vectorizer.pkl')
        self.metrics_path = os.path.join(self.base_dir, 'model', 'model_metrics.json')

        self.model = None
        self.vectorizer = None
        self.metrics = {}
        self.model_name = "Unknown Model"

        self.load_model_artifacts()

    def load_model_artifacts(self):
        """Loads trained machine learning model and TF-IDF vectorizer."""
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, 'r', encoding='utf-8') as f:
                    self.metrics = json.load(f)
                    self.model_name = self.metrics.get('best_model', 'Trained Classifier')
        else:
            self.model = None
            self.vectorizer = None
            self.model_name = "Model Not Trained"

    def is_ready(self):
        return self.model is not None and self.vectorizer is not None

    def detect_category(self, text):
        """Detects sub-category hints based on keywords if cyberbullying is present."""
        lower = text.lower()
        if any(w in lower for w in ['female', 'girl', 'woman', 'women', 'blonde', 'kitchen']):
            return 'Gender Cyberbullying'
        elif any(w in lower for w in ['immigrant', 'foreigner', 'race', 'ethnic', 'country', 'alien', 'village']):
            return 'Ethnicity/Racial Cyberbullying'
        elif any(w in lower for w in ['god', 'religion', 'church', 'mosque', 'temple', 'faith', 'pray', 'worship']):
            return 'Religious Cyberbullying'
        elif any(w in lower for w in ['old', 'boomer', 'dinosaur', 'grandma', 'grandpa', 'fossil', 'retire']):
            return 'Age Cyberbullying'
        return 'General Toxicity'

    def predict(self, raw_text):
        if not raw_text or not raw_text.strip():
            return {
                'error': 'Text input cannot be empty.',
                'status': 'error'
            }

        # Auto-reload if not loaded yet
        if not self.is_ready():
            self.load_model_artifacts()
            if not self.is_ready():
                return {
                    'error': 'Model has not been trained yet. Please run python train_model.py first.',
                    'status': 'error'
                }

        # Step 1: Preprocess input text
        cleaned = clean_text(raw_text)
        
        # Fallback if cleaning removes everything (e.g. only symbols)
        if not cleaned.strip():
            cleaned = raw_text.strip().lower()

        # Step 2: Vectorize using trained TF-IDF
        features = self.vectorizer.transform([cleaned])

        # Step 3: Classify using trained Machine Learning Model
        pred_class = int(self.model.predict(features)[0])

        # Step 4: Calculate Confidence Score
        confidence = 0.50
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(probabilities[pred_class])
        elif hasattr(self.model, 'decision_function'):
            decision_val = float(self.model.decision_function(features)[0])
            # Sigmoid conversion for SVM / Decision Function outputs
            confidence = 1.0 / (1.0 + np.exp(-abs(decision_val)))
            if pred_class == 0:
                confidence = float(np.clip(confidence, 0.55, 0.99))
            else:
                confidence = float(np.clip(confidence, 0.60, 0.99))
        else:
            confidence = 0.85

        # Format confidence as percentage (50.0% to 99.9%)
        conf_percentage = round(confidence * 100, 1)

        is_cyberbullying = (pred_class == 1)
        label_str = "Cyberbullying" if is_cyberbullying else "Not Cyberbullying"

        # Severity ranking
        if not is_cyberbullying:
            severity = "Safe"
            category = "Clean Content"
        else:
            category = self.detect_category(raw_text)
            if conf_percentage >= 85:
                severity = "High"
            elif conf_percentage >= 70:
                severity = "Medium"
            else:
                severity = "Low"

        return {
            'status': 'success',
            'raw_text': raw_text,
            'cleaned_text': cleaned,
            'prediction': label_str,
            'is_cyberbullying': is_cyberbullying,
            'confidence': conf_percentage,
            'severity': severity,
            'category': category,
            'model_name': self.model_name
        }

# Global predictor instance
predictor = CyberbullyingPredictor()
