# 🛡️ Professional Email Spam Classifier

An end-to-end Machine Learning pipeline designed to classify emails as **Spam** or **Ham** using a classification algorithm (KNN, Logistic Regression or Naive Bayes) and Natural Language Processing (NLP).

## 🚀 Overview

This project transitions from basic scripts to a professional ML pipeline. It utilizes **TF-IDF Vectorization** with N-gram context and evaluates multiple classification algorithms, ultimately deploying the most robust model via **Streamlit**.

## 🛠️ Technical Stack

- **Languages:** Python 3.10+
- **Hardware:** Developed on MSI Sword 16 HX (i7-13700HX / RTX 4060)
- **Libraries:** Scikit-Learn, NLTK, Pandas, NumPy
- **NLP Techniques:** Regex Cleaning, Lemmatization, Stop-word Removal, N-grams (1, 2)
- **Deployment:** Streamlit Cloud / Localhost

## 📊 The ML Pipeline

1. **Data Cleaning:** Regex-based removal of non-alpha characters and noise (e.g., 'escapenumber').
2. **Advanced NLP:** Lemmatization via `WordNetLemmatizer` to reduce vocabulary dimensionality.
3. **Feature Engineering:** TF-IDF Vectorization with `max_features=15000` and `ngram_range=(1, 2)`.
4. **Model Training:**
   - **Logistic Regression (Winner):** Optimized with `C=0.5` for high-dimensional generalization.
   - **Multinomial Naive Bayes:** Baseline for text classification.
   - **K-Nearest Neighbors:** Sampled approach (10k rows) for computational efficiency.
5. **Deployment:** Real-time inference engine built with Streamlit.

## 📈 Model Comparison

| Model                   | Accuracy | Precision |  Recall  | F1-Score |
| :---------------------- | :------: | :-------: | :------: | :------: |
| **Logistic Regression** | **~98%** | **~98%**  | **~97%** | **~98%** |
| Naive Bayes             |   ~96%   |   ~94%    |   ~98%   |   ~96%   |
| KNN (Sampled)           |   ~75%   |   ~72%    |   ~81%   |   ~76%   |

## 🧠 Explainable AI (XAI) Case Study

During testing, a formal leave application was flagged as spam due to **Dataset Bias**. We developed a debugger script to analyze the model's internal weights:

- **Spam Triggers:** Formalisms like "Dear Sir" (+1.27) and "Sincerely" (+0.54).
- **Safe Indicators:** Specificity like dates ("April" -2.38) and names ("John" -1.95).
- **Optimization:** We successfully calibrated the system by adjusting the **Probability Threshold to 75%**, prioritizing Precision to eliminate False Positives.

## 🏃 How to Run

1. Clone the git repository

   ```bash
    git clone https://github.com/premanshtripathi/spam-email-classifier.git
   ```

2. Ensure `spam_model_lr.pkl` and `tfidf_vectorizer.pkl` are in the root directory.
3. Install dependencies: `pip install streamlit scikit-learn nltk pandas`
4. Launch the app:
   ```bash
   streamlit run app.py
   ```

---
