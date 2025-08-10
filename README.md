# Fake-Review-Detection.
# Fake Review Detection Web App

A Flask-based web application that classifies product reviews as **Genuine** or **Fake** using a combination of TF-IDF and custom text-derived features.

---

## Table of Contents
- [Overview](#overview)  
- [Project Structure](#project-structure)  
- [Setup & Usage](#setup--usage)  
- [How It Works](#how-it-works)  
- [UI Highlights](#ui-highlights)  
- Explore our analysis and modeling steps in the [project notebook (pro1.ipynb)](pro1.ipynb). 
- [Future Enhancements](#future-enhancements)  
- [Contributing & License](#contributing--license)

---

## Overview

Paste a product review into the input field and get a prediction—**Genuine** or **Fake**—with an associated probability score. Model files (`.pkl`) are efficiently tracked with Git LFS.

---

## Project Structure

FakeReviewDetection/
├── model_files/
│ ├── fake_review_model.pkl
│ ├── tfidf_vectorizer.pkl
│ └── scaler.pkl
├── static/
│ └── style.css
├── templates/
│ └── index.html
├── app.py
├── requirements.txt
├── fake_review_notebook.ipynb
└── README.md

yaml
Copy
Edit

---

## Setup & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Harshayellapu/Fake-Review-Detection.git
   cd Fake-Review-Detection
pip install -r requirements.txt
python app.py

How It Works
clean_text() lowercases the text, removes URLs, punctuation, and extra whitespace.

extract_custom_features() calculates features like word count, character count, uppercase letters, exclamation marks, and spam keywords.

The prediction pipeline:

Cleaned text → TF-IDF vector

Extract custom features → scaling

Combine these inputs → model predicts probability

Prob ≥ 90% → Genuine, else Fake

UI Highlights
Modern, responsive design with smooth animations.

Clear form with input area and dynamic results.

Badges are color-coded: green = Genuine, red = Fake.

Sticky header and footer for polished user experience.

Explore the Notebook (Optional)
For detailed analysis, modeling process, and results, check out:

fake_review_notebook.ipynb

Future Enhancements
Add advanced NLP models like BERT or Transformer-based classifiers.

Improve UI with feedback, loading indicators, or interactive charts.

Deploy via Docker, Heroku, or other cloud platforms.

