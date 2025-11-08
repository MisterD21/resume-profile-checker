📄 AI Resume Classifier (Streamlit + NLP + ML)

This project is an AI-powered Resume Classification App built using Streamlit, NLP, and Machine Learning (KNN).
It analyzes the content of resumes, cleans the text, and predicts the most relevant job category (like Data Science, Web Developer, HR, etc.) based on training data.

🚀 Features

✅ Upload your resume (PDF format)
✅ Automatically extract and clean resume text
✅ Classify the resume into a job category using a trained ML model
✅ Uses TF-IDF vectorization + K-Nearest Neighbors (KNN) for prediction
✅ Simple and interactive Streamlit web app interface

🧩 How It Works
	1.	Upload Resume
The user uploads a resume in PDF format.
	2.	Extract Text
Text is extracted from the PDF using pdfplumber.
	3.	Clean Resume Text
The text is cleaned using regex — removing URLs, hashtags, mentions, punctuation, and stopwords.
	4.	Model Training
	•	The app loads the UpdatedResumeDataSet.csv dataset.
	•	Applies TF-IDF vectorization to convert text into numerical form.
	•	Trains a KNN classifier using the cleaned data and corresponding job categories.
	5.	Prediction
When a new resume is uploaded, it is transformed using the same TF-IDF model, and the app predicts its category label.
	6.	Output
The predicted category is displayed in the Streamlit interface.


📁 Project Structure

📂 AI-Resume-Classifier
│
├── files/
│   └── UpdatedResumeDataSet.csv     # Training dataset
│
├── streamlit_app.py                 # Main application script
│
├── requirements.txt                 # Dependencies for deployment
│
└── README.md                        # Project documentation

⚙️ Installation and Setup

1️⃣ Clone the Repository

git clone https://github.com/MisterD21/resume-profile-checker.git
cd ai-resume-classifier

python -m venv .venv
source .venv/bin/activate   # for macOS/Linux
.venv\Scripts\activate      # for Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the App
streamlit run streamlit_app.py

🧪 Example Output

When you upload a resume, you’ll see:
✅ Resume processed successfully!

Predicted Job Category:
Data Science

💡 Future Enhancements

🔹 Add ATS Score Checker for job description matching
🔹 Integrate Named Entity Recognition (NER) using SpaCy
🔹 Show top 3 matching roles instead of one
🔹 Deploy to Streamlit Cloud or Hugging Face Spaces

⸻

🧑‍💻 Author

Nandan Dubey
💼 Java & Web Developer | AI Enthusiast
📧 nandandubey44@gmail.com
🌐 https://www.linkedin.com/in/nandandubey44/


