import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Fallback English stop words set (no internet required)
DEFAULT_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't",
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't",
    'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during',
    'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he',
    "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's",
    'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's",
    'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or',
    'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll",
    "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs',
    'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've",
    'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd",
    "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which',
    'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd",
    "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'
}

import nltk
try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = DEFAULT_STOPWORDS

try:
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
except Exception:
    lemmatizer = None

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def clean_text(text):
    """
    Comprehensive text cleaning and NLP preprocessing pipeline:
    - Lowercasing
    - Stripping URLs, HTML tags, user mentions (@user)
    - Handling hashtags (#hashtag -> hashtag)
    - Removing non-alphabetic characters and numbers
    - Tokenization, stopword removal, and lemmatization
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove user mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtag symbol but keep text
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization & Stopword removal & Lemmatization
    tokens = text.split()
    cleaned_tokens = []
    for word in tokens:
        if word not in stop_words and len(word) > 1:
            if lemmatizer:
                try:
                    word = lemmatizer.lemmatize(word)
                except Exception:
                    pass
            cleaned_tokens.append(word)
    
    return ' '.join(cleaned_tokens)


def load_dataset(dataset_path):
    """
    Loads dataset from CSV file and handles flexible column naming formats
    (Kaggle format: 'tweet_text', 'cyberbullying_type' or generic 'text', 'label').
    """
    if not os.path.exists(dataset_path):
        alt_path = os.path.join(os.path.dirname(dataset_path), 'cyberbullying_tweets.csv')
        if os.path.exists(alt_path):
            dataset_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"[*] Successfully loaded dataset from {dataset_path} ({len(df)} records)")

    # Detect text column name
    text_col = None
    for candidate in ['tweet_text', 'text', 'comment', 'message', 'content']:
        if candidate in df.columns:
            text_col = candidate
            break
            
    if not text_col:
        text_col = df.columns[0]

    # Detect label column name
    label_col = None
    for candidate in ['cyberbullying_type', 'label', 'is_cyberbullying', 'cyberbullying', 'target']:
        if candidate in df.columns:
            label_col = candidate
            break
            
    if not label_col:
        label_col = df.columns[1]

    print(f"[*] Using columns: text='{text_col}', label='{label_col}'")

    df = df.dropna(subset=[text_col, label_col])

    # Standardize label into binary (0 = Not Cyberbullying, 1 = Cyberbullying)
    def normalize_label(val):
        str_val = str(val).strip().lower()
        if str_val in ['0', 'not_cyberbullying', 'non-cyberbullying', 'clean', 'normal', 'false', 'safe']:
            return 0
        return 1

    df['binary_label'] = df[label_col].apply(normalize_label)
    df['raw_text'] = df[text_col]
    
    if label_col == 'cyberbullying_type':
        df['category'] = df[label_col].astype(str)
    else:
        df['category'] = df['binary_label'].map({0: 'not_cyberbullying', 1: 'cyberbullying'})

    return df


def train_and_evaluate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(base_dir, 'dataset', 'cyberbullying_dataset.csv')
    model_dir = os.path.join(base_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    print("==================================================")
    print("      CYBERBULLYING DETECTION MODEL TRAINING      ")
    print("==================================================")

    # 1. Load Data
    df = load_dataset(dataset_file)
    
    # 2. Text Preprocessing
    print("[*] Preprocessing text data with NLTK pipeline...")
    df['cleaned_text'] = df['raw_text'].apply(clean_text)
    
    # Drop empty preprocessed rows
    df = df[df['cleaned_text'].str.strip() != '']
    
    X = df['cleaned_text']
    y = df['binary_label']
    
    print(f"[*] Total valid processed samples: {len(df)}")
    print(f"    - Cyberbullying samples (1): {sum(y == 1)}")
    print(f"    - Non-Cyberbullying samples (0): {sum(y == 0)}")

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    # 4. Feature Extraction (TF-IDF)
    print("[*] Extracting features using TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1 if len(df) < 500 else 2,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"[*] TF-IDF Matrix shape: {X_train_tfidf.shape}")

    # 5. Define Candidate Models
    candidate_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        'Naive Bayes': MultinomialNB(alpha=0.1),
        'Linear SVM': LinearSVC(C=1.0, max_iter=2000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    }

    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_model_obj = None

    print("\n[*] Training and comparing models:")
    print("-" * 75)
    print(f"{'Model Name':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 75)

    for name, model in candidate_models.items():
        # Train
        model.fit(X_train_tfidf, y_train)
        
        # Predict
        y_pred = model.predict(X_test_tfidf)
        
        # Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'confusion_matrix': cm
        }

        print(f"{name:<22} | {acc*100:6.2f}%    | {prec*100:6.2f}%    | {rec*100:6.2f}%    | {f1*100:6.2f}%")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model

    print("-" * 75)
    print(f"\n[+] BEST MODEL SELECTED: {best_model_name} (F1-Score: {best_f1*100:.2f}%)")

    # 6. Save Artifacts
    model_path = os.path.join(model_dir, 'cyberbullying_model.pkl')
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    metrics_path = os.path.join(model_dir, 'model_metrics.json')

    joblib.dump(best_model_obj, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    metrics_summary = {
        'best_model': best_model_name,
        'trained_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'vocab_size': len(vectorizer.vocabulary_),
        'models_performance': results
    }

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=4)

    print(f"[+] Saved trained model to: {model_path}")
    print(f"[+] Saved vectorizer to: {vectorizer_path}")
    print(f"[+] Saved model metrics to: {metrics_path}")
    print("==================================================\n")

if __name__ == '__main__':
    train_and_evaluate()
