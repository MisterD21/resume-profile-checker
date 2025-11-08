import streamlit as st
import pdfplumber
import re
from nltk.corpus import stopwords
import os
import nltk
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# 🧹 Cleaning function
def clean_resume(text):
    text = re.sub(r'http\S+\s*', ' ', text)  # remove URLs
    text = re.sub(r'RT|cc', ' ', text)       # remove RT and cc
    text = re.sub(r'@\S+', ' ', text)        # remove mentions
    text = re.sub(r'#\S+', ' ', text)        # remove hashtags
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)  # keep only letters/numbers
    words = text.lower().split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# 🧠 Streamlit UI
st.title("📄 AI Resume Text Extractor")

uploaded_file = st.file_uploader("Upload your resume (PDF format)", type=["pdf"])

if uploaded_file is not None:

    
    df = pd.read_csv('files/UpdatedResumeDataSet.csv')
    df['Resume'].apply(lambda x: clean_resume(x))
    le = LabelEncoder()
    le.fit(df['Category'])
    df['Category'] = le.transform(df['Category'])
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf.fit(df['Resume'])
    requiredText = tfidf.transform(df['Resume'])
    X_train, X_test, y_train, y_test = train_test_split(requiredText, df['Category'], test_size=0.33, random_state=42)   
    knn = OneVsRestClassifier(KNeighborsClassifier())
    knn.fit(X_train, y_train)
    ypred = knn.predict(X_test)


    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""

    # st.subheader("🧾 Extracted Text")
    # st.text_area("Resume Content", text, height=200)

    cleaned = clean_resume(text)
    # st.subheader("✨ Cleaned Resume Text")
    # st.text_area("Cleaned Content", cleaned, height=200)

    predicted_label = knn.predict(tfidf.transform([cleaned]))
    predicted_label_num = predicted_label[0]
    predicted_label_name = le.inverse_transform([predicted_label_num])[0]
    st.text_area("predicted_label_name", predicted_label_name, height=10)


    # You can add AI analysis or skill extraction here
    st.success("✅ Resume processed successfully!")
else:
    st.info("👆 Please upload a PDF to get started.")