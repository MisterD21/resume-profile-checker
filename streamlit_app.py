import streamlit as st
import pdfplumber
import re
from nltk.corpus import stopwords
import nltk

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
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""

    st.subheader("🧾 Extracted Text")
    st.text_area("Resume Content", text, height=200)

    cleaned = clean_resume(text)
    st.subheader("✨ Cleaned Resume Text")
    st.text_area("Cleaned Content", cleaned, height=200)

    # You can add AI analysis or skill extraction here
    st.success("✅ Resume processed successfully!")
else:
    st.info("👆 Please upload a PDF to get started.")