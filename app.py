from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# -----------------------------
# Load Model, Vectorizer, Scaler
# -----------------------------
with open("model_files/fake_review_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_files/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model_files/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# Text Cleaning and Features
# -----------------------------
spam_keywords = ['buy now', 'click here', 'subscribe', 'limited offer',
                 'free', 'money back', 'hurry', 'guarantee', 'order now',
                 'deal', 'get rich', 'amazing', 'best product']

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_custom_features(text):
    text_lower = text.lower()
    return pd.Series({
        'word_count': len(text.split()),
        'char_count': len(text),
        'uppercase_count': sum(1 for w in text.split() if w.isupper()),
        'exclamation_count': text.count('!'),
        'spam_words_count': sum(1 for word in spam_keywords if word in text_lower)
    })

# -----------------------------
# Flask Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    cleaned = clean_text(review)
    tfidf_vec = tfidf.transform([cleaned]).toarray()
    
    custom_feat = pd.DataFrame([extract_custom_features(cleaned)])
    scaled_custom = scaler.transform(custom_feat)

    X_input = np.hstack((tfidf_vec, scaled_custom))

    prob = model.predict_proba(X_input)[0][1]
    label = "Genuine" if prob >= 0.90 else "Fake"

    return render_template('index.html', prediction=label, probability=f"{prob:.2%}")

# -----------------------------
# Run the App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
