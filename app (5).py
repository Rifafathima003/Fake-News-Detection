import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------
# Download NLTK data (cached, runs once)
# -------------------------------
@st.cache_resource
def setup_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()

stop_words, lemmatizer = setup_nltk()

# -------------------------------
# Load model + vectorizer (cached)
# -------------------------------
@st.cache_resource
def load_artifacts():
    try:
        with open("logistic_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"❌ Required file not found: {e.filename}. "
                 "Make sure logistic_model.pkl and tfidf_vectorizer.pkl are in the app directory.")
        st.stop()

model, vectorizer = load_artifacts()

# -------------------------------
# Preprocessing
# -------------------------------
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

st.title("📰 Fake News Detector (Logistic Regression)")
st.write(
    "This app uses **Machine Learning** with TF-IDF + Logistic Regression "
    "to classify news as **Real ✅** or **Fake ❌**."
)

default_text = """Donald Trump met with reporters at the White House 
and announced a new healthcare policy on Tuesday."""

user_input = st.text_area("✍️ Enter a news article below:", default_text, height=150)

if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to classify.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        if prediction == 1:
            st.error("🟥 FAKE News Detected")
        else:
            st.success("🟩 REAL News Detected")

st.markdown("---")
st.markdown("👩‍💻 *Built with Streamlit | Logistic Regression | TF-IDF*")
