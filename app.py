import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1. Download necessary NLTK data (Streamlit will handle this silently)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# 2. Initialize our NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# 3. The exact same cleaning function from our Jupyter Notebook
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = text.replace('escapenumber', '')
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# 4. Load the Pickled Model and Vectorizer
# Streamlit's cache decorator prevents it from reloading the file every time you click a button
@st.cache_resource 
def load_models():
    model = joblib.load('spam_model_lr.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_models()

# ==========================================
# 5. Streamlit User Interface (UI)
# ==========================================

# Page Configuration
st.set_page_config(page_title="Spam Classifier", page_icon="🛡️")

st.title("🛡️ Email Spam Classifier")
st.markdown("""
Welcome to the Spam Email Classifier.
""")
st.markdown("""
Enter the text of an email below to check if it's **Spam** or **Safe (Ham)**.
""")

# Text input box for the user
user_input = st.text_area("Paste email content here:", height=200, placeholder="Dear customer, you have won a free prize...")

# The prediction button
if st.button("Analyze Email", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text using Logistic Regression..."):
            
            # Step A: Clean the raw input
            cleaned_input = clean_text(user_input)
            
            # Step B: Convert the cleaned text into numerical vectors
            vectorized_input = vectorizer.transform([cleaned_input])
            
            # Step C: Get the EXACT probability percentages instead of a simple 0 or 1
            # [0] is the chance it's Ham, [1] is the chance it's Spam
            probabilities = model.predict_proba(vectorized_input)[0]
            spam_probability = probabilities[1]
            
            # Display Results
            st.divider()
            
            # We are manually raising the bar. It must be > 75% confident to call it spam.
            if spam_probability > 0.75:
                st.error(f"🚨 **WARNING: SPAM DETECTED** (Confidence: {spam_probability*100:.1f}%)")
            else:
                st.success(f"✅ **SAFE: HAM** (Spam Probability was only {spam_probability*100:.1f}%)")
            

            # # Step C: Make the prediction
            # prediction = model.predict(vectorized_input)[0]
            
            # # Display Results
            # st.divider()
            # if prediction == 1:
            #     st.error("🚨 **WARNING: This email is classified as SPAM.**")
            # else:
            #     st.success("✅ **SAFE: This email is classified as HAM (Not Spam).**")
