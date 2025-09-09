import streamlit as st
import pickle
import re
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------
# Load model + vectorizer (Pickle)
# -------------------------------
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -------------------------------
# Preprocessing setup
# -------------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Basic preprocessing: remove non-letters, lowercase, remove stopwords, lemmatize"""
    text = re.sub(r'[^a-zA-Z]', ' ', text)  
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return " ".join(text)

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Title + description
st.title("📰 Fake News Detector (Logistic Regression)")
st.write("This app uses **Machine Learning** with TF-IDF + Logistic Regression "
         "to classify news as **Real ✅** or **Fake ❌**.")

# Example text to guide users
default_text = """Donald Trump met with reporters at the White House 
and announced a new healthcare policy on Tuesday."""

# Input text area
user_input = st.text_area("✍️ Enter a news article below:", default_text, height=150)

# Predict button
if st.button("🔍 Predict"):
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]

    if prediction == 1:
        st.error("🟥 FAKE News Detected")
    else:
        st.success("🟩 REAL News Detected")

# Footer
st.markdown("---")
st.markdown("👩‍💻 *Built with Streamlit | Logistic Regression | TF-IDF*")
    # paste your streamlit code here
    
