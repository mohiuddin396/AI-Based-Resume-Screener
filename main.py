import os
import PyPDF2
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data (first time only)
nltk.download('punkt')
nltk.download('stopwords')

# Streamlit Page Config
st.set_page_config(page_title="AI BASED RESUME SCREENER", page_icon="logo.png", layout="wide")

# Custom Styles (Updated with color change for white elements)
st.markdown(
    """
<style>
    body { 
        background-color: #1f2833; /* Dark blue-gray */
        color: #d3e7bb; /* Light green */
        font-family: Arial, sans-serif; 
        margin: 0; 
        padding: 0; 
        position: relative; 
        min-height: 100vh;
    }
    .header {
        background: radial-gradient(circle, #1f2833, #746f8a); /* Dark blue-gray to muted purple-gray */
        padding: 20px;
        text-align: center;
        border-radius: 10px 10px 0 0;
        margin-bottom: 20px;
        position: relative;
        z-index: 1;
    }
    .header h1 { 
        font-size: 80px; 
        font-weight: bold; 
        color: #d3e7bb; /* Light green */
        margin: 0;
        font-family: 'Dancing Script', cursive;
    }
    .header span { 
        font-size: 36px; 
        color: #ecb04b; /* Warm yellow */
    }
    .stTextInput, .stNumberInput, .stFileUploader { 
        background-color: #746f8a !important; /* Muted purple-gray */
        color: #d3e7bb !important; /* Light green */
        border: 1px solid #1f2833; /* Dark blue-gray */
        border-radius: 8px; 
        margin-bottom: 15px;
        padding: 10px;
        position: relative;
        z-index: 1;
    }
    .stTextInput>div>input, .stNumberInput>div>input { color: #d3e7bb !important; }
    .stFileUploader label { color: #d3e7bb !important; }
    .stButton>button { 
        background-color: #ecb04b; /* Warm yellow */
        color: #1f2833; /* Dark blue-gray */
        border-radius: 8px; 
        font-size: 18px; 
        font-weight: bold; 
        padding: 10px 20px; 
        width: 100%;
        margin-top: 10px;
        position: relative;
        z-index: 1;
    }
    .stButton>button:hover { 
        background-color: #d3e7bb; /* Light green */
        transform: scale(1.05); 
        transition: 0.3s ease-in-out; 
    }
    .custom-label { font-size: 16px; color: #d3e7bb; margin-bottom: 5px; }
    .custom-success { font-size: 16px; color: #1f2833; background-color: #d3e7bb; padding: 10px; border-radius: 8px; margin-top: 10px; }
    .stFileUploader>div>div { 
        background-color: #746f8a !important; /* Muted purple-gray */
        border: 2px dashed #ecb04b !important; /* Warm yellow */
        border-radius: 8px; 
        color: #d3e7bb; 
        padding: 20px;
        text-align: center;
        position: relative;
        z-index: 1;
    }
    .uploadedFile { background-color: #746f8a !important; color: #d3e7bb !important; border: 1px solid #1f2833 !important; border-radius: 8px; margin-top: 10px; }
    .stApp { 
        max-width: 1000px; 
        margin: 0 auto; 
        padding: 20px; 
        background-color: transparent; 
        border-radius: 20px; 
        box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.3), 
                    -5px -5px 15px rgba(255, 255, 255, 0.1);
        margin-top: 80px; 
        margin-bottom: 20px;
    }
</style>
    """,
    unsafe_allow_html=True,
)

# Paths
RESUME_FOLDER_PATH = "E:/Resume_Screener/resumes"

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers, but keep job-specific terms
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Custom stopwords (excluding job-specific terms)
    stop_words = set(stopwords.words('english')) - {
        'android', 'developer', 'software', 'engineer', 'data', 'scientist', 
        'ui', 'ux', 'designer', 'python', 'java', 'javascript', 'sql', 'cloud', 
        'aws', 'azure', 'machine', 'learning', 'ai', 'frontend', 'backend', 'fullstack'
    }
    tokens = [word for word in tokens if word not in stop_words]
    # Join back to string
    return ' '.join(tokens)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_source):
    text = ""
    reader = PyPDF2.PdfReader(pdf_source)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return preprocess_text(text)

# Function to calculate similarity score with scaling
def calculate_similarity(job_role, resume_texts):
    # Preprocess job role
    job_role = preprocess_text(job_role)
    # Vectorizer with tuned parameters
    vectorizer = TfidfVectorizer(
        max_df=0.9,            # Increased to capture more terms
        min_df=1,              # Keep terms appearing at least once
        ngram_range=(1, 5),    # Increased range to capture longer phrases
        stop_words=None,       # Stopwords handled in preprocess_text
        max_features=15000,    # Increased vocabulary size
        sublinear_tf=True      # Apply sublinear term frequency scaling for better distribution
    )
    # Combine job role and resume texts
    documents = [job_role] + resume_texts
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Log raw cosine similarities for debugging
    # st.write("Raw Cosine Similarities:", cosine_similarities)
    
    # Dynamic scaling based on the maximum raw similarity
    max_similarity = np.max(cosine_similarities) if np.max(cosine_similarities) > 0 else 1
    scaling_factor = 1.0 / max_similarity if max_similarity > 0 else 1.0  # Normalize to max similarity
    scaled_similarities = cosine_similarities * scaling_factor * 0.95  # Scale to 95% max
    scaled_similarities = np.clip(scaled_similarities, 0, 1)  # Cap at 1 (100%)
    return scaled_similarities

# Streamlit UI
st.markdown("<div class='header'><h1><span>AI-Based Resume Screener</span></h1></div>", unsafe_allow_html=True)

st.markdown("<p class='custom-label'>Enter Job Role<br><small style='color: #d3e7bb;'>Examples: UI Designer, Software Developer, Data Scientist...</small></p>", unsafe_allow_html=True)
job_role = st.text_input("", key="job_input", value="Android Developer")

st.markdown("<p class='custom-label'>Number of Top Candidates to Display<br><small style='color: #d3e7bb;'>Select how many candidates you want to see in results.</small></p>", unsafe_allow_html=True)
top_n = st.number_input("", min_value=1, value=2, step=1)

st.markdown("<p class='custom-label'>Upload Resume(s) (PDF)</p>", unsafe_allow_html=True)
uploaded_files = st.file_uploader("", type=["pdf"], accept_multiple_files=True, key="upload_box")

if uploaded_files:
    st.markdown(f"<p class='custom-success'>✅ {len(uploaded_files)} Resume(s) Uploaded Successfully!</p>", unsafe_allow_html=True)

if st.button("**Screen Resumes**", key="search_btn"):
    if not job_role:
        st.warning("Please enter a job role.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        resume_texts = []
        for uploaded_file in uploaded_files:
            resume_text = extract_text_from_pdf(uploaded_file)
            resume_texts.append(resume_text)

        similarity_scores = calculate_similarity(job_role, resume_texts)

        results = list(zip([uploaded_file.name for uploaded_file in uploaded_files], similarity_scores))
        # Filter results with threshold (5% or higher)
        results = [(file_name, score) for file_name, score in results if score >= 0.05]
        results.sort(key=lambda x: x[1], reverse=True)

        if results:
            st.markdown("<h3 style='font-size: 20px; color: #d3e7bb;'>Top Employees:</h3>", unsafe_allow_html=True)
            for i, (file_name, score) in enumerate(results[:top_n], start=1):
                st.markdown(f"<p style='font-size: 16px; color: #d3e7bb;'>{i}. {file_name} - Similarity Score: {score * 100:.2f}%</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-size: 16px; color: #d3e7bb;'>No candidates match the Job category.</p>", unsafe_allow_html=True)