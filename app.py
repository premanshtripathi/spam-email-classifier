import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_


import streamlit as st
import torch
import torch.nn as nn
import joblib
from keras_preprocessing.sequence import pad_sequences

# ==========================================
# 1. Model Architecture Define Karo (Vahi same class)
# ==========================================
class SpamLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_units):
        super(SpamLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_units, batch_first=True)
        self.fc = nn.Linear(lstm_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]
        x = self.fc(x)
        return self.sigmoid(x)

# ==========================================
# 2. Load the PyTorch Model & Metadata
# ==========================================
@st.cache_resource 
def load_pytorch_artifacts():
    # Metadata load karo (Tokenizer, max_len, etc.)
    meta = joblib.load('model_metadata.pkl')
    
    # Model initialize karo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpamLSTM(meta['vocab_size'], meta['embed_dim'], meta['lstm_units'])
    
    # Weights load karo
    model.load_state_dict(torch.load('spam_lstm_model.pth', map_location=device))
    model.to(device)
    model.eval() # Evaluation mode ON!
    
    return model, meta, device

model, meta, device = load_pytorch_artifacts()

# ==========================================
# 3. Streamlit UI
# ==========================================
st.set_page_config(page_title="LSTM Spam Classifier", page_icon="🤖")
st.title("🛡️ LSTM Email Spam Classifier")
st.markdown("Trained on the dataset of 83k+ Rows!")

user_input = st.text_area("Paste email content here:", height=200, placeholder="Win a free gift card...")

if st.button("Analyze Email", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("LSTM processing..."):
            
            # Step A: Preprocessing (Same as training)
            # Agar training mein NLTK cleaning ki thi, toh yahan bhi clean_text(user_input) call karein
            sequences = meta['tokenizer'].texts_to_sequences([user_input])
            padded = pad_sequences(sequences, maxlen=meta['max_len'])
            
            # Step B: Tensor Conversion
            tensor_input = torch.from_numpy(padded).long().to(device)
            
            # Step C: Prediction
            with torch.no_grad(): # No gradient needed for inference
                output = model(tensor_input)
                spam_probability = output.item()
            
            # Display Results
            st.divider()
            if spam_probability > 0.70: # 70% threshold
                st.error(f"🚨 **SPAM DETECTED** (Confidence: {spam_probability*100:.2f}%)")
            else:
                st.success(f"✅ **SAFE: HAM** (Spam Probability: {spam_probability*100:.2f}%)")