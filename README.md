# Fake-News-Detection

# Fake News Detection using Logistic Regression

This project focuses on detecting Fake News using Machine Learning techniques. The model is built on a Logistic Regression classifier with TF-IDF vectorization for feature extraction. 
A Streamlit Web App is deployed to allow users to test news articles and classify them as Real or Fake.

## 🔍 Features
- Preprocessing & TF-IDF vectorization
- Model training and evaluation (Accuracy ~99%)
- Classifier comparison (SVM, Naive Bayes, Neural Net)
- Deployed web app for live predictions

## 🚀 Try the App
👉 [Click here to view the app](https://fake-news-detector-ensknseqzfvcv8p87pgvo4.streamlit.app/)

## 📁 Files
- `app.py` — Streamlit frontend
- `logistic_model.pkl` — Trained classifier
- `tfidf_vectorizer.pkl` — TF-IDF feature transformer
- `requirements.txt` — Dependencies

## 📊 Sample Input
> "NASA successfully launches Artemis mission to the moon."

→ Output: ✅ Real News

## 📚 References
- Used dataset: [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data)
- Literature survey: Based on 4 published papers
