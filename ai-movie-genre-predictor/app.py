import streamlit as st
import pandas as pd
import re
import joblib

# 🎯 Load trained model
model = joblib.load("genre_model.pkl")

# 🧼 Text cleaner
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation/numbers
    text = text.lower()
    return text

# 🎨 Streamlit page setup
st.set_page_config(page_title="🎬 Movie Genre Predictor", layout="centered")

# 💅 Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #fffafc;
    }
    .big-title {
        font-size: 2.5em;
        font-weight: bold;
        color: #ff4b91;
    }
    .predicted {
        font-size: 1.3em;
        font-weight: bold;
        color: #6a0572;
    }
    .stTextArea > div > textarea {
        color: black !important;
        background-color: #fff5fa !important;
    }
    </style>
""", unsafe_allow_html=True)

# 🖼️ Title
st.markdown('<div class="big-title">🎥 AI Movie Genre Predictor</div>', unsafe_allow_html=True)
st.write("Give your movie description, and I’ll guess the genre like a true cinephile 🎬✨")

# 📝 Input
desc = st.text_area("📜 Movie Description", height=200, placeholder="A boy discovers a magical school...")

# 🧠 Predict genre
if st.button("🔮 Predict Genre"):
    if desc.strip():
        cleaned = clean_text(desc)
        predicted_genre = model.predict([cleaned])[0]
        st.markdown(f'<div class="predicted">🎭 Predicted Genre: <span style="color:#ff4b91;">{predicted_genre}</span></div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter a movie description first!")
