import streamlit as st
import pickle
import re
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the pre-trained model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return cleaned_tokens

def predict_sentiment(text):
    cleaned_text = clean_text(text)
    tokens = tokenize_and_lemmatize(cleaned_text)
    lemmatized_text = ' '.join(tokens)
    features = vectorizer.transform([lemmatized_text])
    prediction = model.predict(features)
    return 'positive' if prediction[0] == 1 else 'negative'

# File paths
history_file = 'history.jsonl'  # Change extension to .jsonl for JSON Lines format

def load_history():
    """Load history from JSON Lines file."""
    history = []
    try:
        with open(history_file, 'r') as f:
            for line in f:
                history.append(json.loads(line.strip()))
        return history, len(history)
    except (FileNotFoundError, json.JSONDecodeError):
        return [], 0

def save_history_item(item):
    """Save a single history item to JSON Lines file."""
    with open(history_file, 'a') as f:
        f.write(json.dumps(item) + '\n')  # Append item as a new line

# Load existing history and count
history, count = load_history()

# Streamlit app interface
st.title("Sentiment Analysis")

user_input = st.text_area("Enter text for sentiment analysis:", "")

if st.button("Analyze"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"Sentiment: **{sentiment}**")
        
        # Update history
        count += 1
        history_item = {
            'id': count,
            'text': user_input,
            'sentiment': sentiment
        }
        history.append(history_item)
        
        # Save the new history item
        save_history_item(history_item)
    else:
        st.write("Please enter text to analyze.")

# Display history
st.write("### Sentiment Analysis History")
if history:
    for item in history:
        # Safeguard to ensure all required keys exist
        if 'id' in item and 'text' in item and 'sentiment' in item:
            st.write(f"**{item['id']}:** {item['text']} - Sentiment: **{item['sentiment']}**")
else:
    st.write("No history available.")
