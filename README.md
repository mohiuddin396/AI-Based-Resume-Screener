# 📄 AI-Based Resume Screener

An AI-powered Resume Screening System built using NLP and Machine Learning techniques. This tool helps HR professionals efficiently filter and rank candidates based on job relevance.

## 🚀 Features
- 📂 Upload multiple resumes (PDF format)
- 🔍 Extract and analyze text using NLP
- 📊 Rank resumes based on job relevance using TF-IDF and cosine similarity
- 📈 Display the top N most suitable candidates
- ⚡ Fast and efficient resume parsing

## 🛠️ Technologies Used
- Python 🐍
- Streamlit 🎨
- Scikit-learn 🤖
- PyPDF2 📑
- NumPy 🔢

## 📌 Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/resume-screener.git
   cd resume-screener
   ```
2. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## 📋 Usage
1. Run the Streamlit application:
   ```sh
   streamlit run main4.py
   ```
2. Enter the job role and upload resumes.
3. Click 'Search Resumes' to rank candidates.
4. View the top candidates based on relevance.

## 🎯 How It Works
- Extracts text from uploaded PDFs.
- Uses TF-IDF vectorization to analyze job descriptions.
- Computes similarity scores between job roles and resumes.
- Displays the top N most relevant resumes.

## 📝 Future Enhancements
- ✅ Support for more file formats (DOCX, TXT)
- 🧠 AI-based skill extraction
- 📅 Integration with job portals

---
⚡ **Contributors:** Your Name  
📧 Contact: your.email@example.com
