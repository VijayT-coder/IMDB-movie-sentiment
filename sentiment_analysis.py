import sys
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"<.*?>", "", text)  # remove HTML tags
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)  # remove numbers
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Load dataset
try:
    df = pd.read_csv("IMDB Dataset.csv")
    print(f"[INFO] Dataset loaded successfully with shape: {df.shape}")
except FileNotFoundError:
    print("[ERROR] Dataset file 'IMDB Dataset.csv' not found.")
    sys.exit(1)

# Clean the text data
df['clean_review'] = df['review'].apply(clean_text)
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split dataset
X = df['clean_review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"[INFO] Model trained. Accuracy on test set: {acc:.4f}")

if len(sys.argv) > 1:
    user_input = " ".join(sys.argv[1:])
    cleaned_input = clean_text(user_input)
    input_vec = vectorizer.transform([cleaned_input])
    prediction = model.predict(input_vec)[0]
    sentiment = "positive" if prediction == 1 else "negative"
    print(f"[RESULT] Predicted Sentiment: {sentiment}")
    sys.exit(0)