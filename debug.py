import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1. Load the saved model and vectorizer
model = joblib.load('spam_model_lr.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# 2. Recreate the cleaning function so we process the text the same way
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = text.replace('escapenumber', '')
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# 3. Paste your specific leave application here
false_positive_email = """
Dear Sir,
I am writing to request a leave of absence from 15 April 2026 to 17 April 2026 due to personal reasons. 
I have updated Jack on my ongoing tasks to ensure a smooth transition. I will complete any urgent work before my departure and will be available over email if anything critical arises. 
Kindly approve my leave request. 
Sincerely,
John Doe
"""

# 4. Clean and Vectorize the email
cleaned_email = clean_text(false_positive_email)
vectorized_email = vectorizer.transform([cleaned_email])

# 5. Extract the "Brain" of the Logistic Regression model
# get_feature_names_out() gets our 5000 vocabulary words
# coef_[0] gets the mathematical weight assigned to each word
words = vectorizer.get_feature_names_out()
weights = model.coef_[0]

# 6. Match the words in YOUR specific email to their weights
# A positive weight pushes the needle towards SPAM (1)
# A negative weight pushes the needle towards HAM (0)
analyzer = vectorizer.build_analyzer()
email_words = analyzer(cleaned_email)
word_weights = []

for word in email_words:
    if word in words: # Check if the word exists in our 5000-word vocabulary
        word_index = vectorizer.vocabulary_[word]
        weight = weights[word_index]
        word_weights.append({'Word': word, 'Spam Weight': weight})

# 7. Display the results in a clean table
df_weights = pd.DataFrame(word_weights)
# Sort by the most "spammy" words at the top
df_weights = df_weights.sort_values(by='Spam Weight', ascending=False).reset_index(drop=True)

print("--- Analysis of False Positive Email ---")
print("Words with a POSITIVE weight act as Spam Triggers.")
print("Words with a NEGATIVE weight act as Safe/Ham indicators.\n")
print(df_weights.to_string())