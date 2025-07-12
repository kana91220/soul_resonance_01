import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.markdown(
    """
    <style>
    .stApp {
        background-color: #fff8fc;
        font-family: 'Comic Sans MS', cursive, sans-serif;
        color: #4b4b4b;
    }

    h1, h2, h3, h4, .stTextInput label, .stSelectbox label,
    .stRadio label, .stSlider label, .stCheckbox label {
        color: #ff66a1 !important;
        font-weight: bold;
    }

    .block-container {
        padding: 2rem;
        border-radius: 20px;
        background-color: #ffe4ec;
        box-shadow: 0 0 15px rgba(255, 182, 193, 0.2);
    }

    /* Buttons */
    .stButton button {
        background-color: #ff99bb;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px 24px;
        border: none;
    }

    /* Sliders */
    .stSlider {
        color: #4b4b4b !important;
    }
    .stSlider > div[data-baseweb="slider"] {
        background-color: #fddde6;
        border-radius: 8px;
        padding: 8px;
    }

    /* Inputs (selectbox, radio, etc.) */
    .stSelectbox div, .stRadio div, .stCheckbox div {
        color: #4b4b4b !important;
        background-color: #fff5f9;
        border-radius: 10px;
    }

    .stSelectbox, .stRadio, .stCheckbox {
        background-color: #fff5f9;
        border-radius: 10px;
        padding: 5px;
    }

    /* Scrollbar (optional glow-up) */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: #ff99bb;
        border-radius: 8px;
    }

)

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("roommates_mock_data.csv")
    return df

df = load_data()

# ------------------ Preprocessing ------------------
features = df[['Wakeup', 'Sleep', 'Cleanliness', 'IntroExtro', 'StudyTime', 'NoiseTolerance', ]]

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), ['Wakeup', 'Sleep', 'StudyTime',]),
    ('num', StandardScaler(), ['Cleanliness', 'IntroExtro', 'NoiseTolerance'])
])

X = preprocessor.fit_transform(features)

# ------------------ Matching Logic ------------------
def find_top_matches(user_input_vector, top_n=3):
    sim_scores = cosine_similarity(user_input_vector, X)[0]
    top_indices = np.argsort(sim_scores)[::-1][:top_n]
    results = [(df.iloc[i]['Name'], round(sim_scores[i]*100, 2)) for i in top_indices]
    return results

# ------------------ Streamlit UI ------------------

st.markdown("<h1>🏠Hostel Roommate Matcher 🌸</h1>", unsafe_allow_html=True)
st.write("Find your perfect roommate match based on lifestyle & personality!")

with st.form("user_input_form"):
    st.subheader("🏨 Tell us about your hostel life vibe:")

    wakeup = st.selectbox("🐓 Usual Wake-up Time", ["🐓 Early (6–8 AM)", "😴 Mid (9–11 AM)", "🦥 Late (12 PM or later)"])
    sleep = st.selectbox("🌙 Sleep Time", ["🌌 Early (Before 11 PM)", "🕰️ Mid (11 PM – 1 AM)", "🌃 Late (2 AM or later)"])
    
    cleanliness = st.slider("🧼 Cleanliness Level", 1, 5, 3,
        help="1 = Chaotic Goblin, 5 = Neat Freak Supreme 🧽")
    
    intro_extro = st.slider("💬 Social Energy", 0.0, 1.0, 0.5,
        help="0 = Ultra introvert 🙈, 1 = Party Animal 🎉")
    
    study_time = st.radio("📚 Study Schedule", ["☀️ Morning", "🌙 Night"])
    
    noise_tolerance = st.slider("🔊 Noise Tolerance", 1, 5, 3,
        help="How much noise can you handle in the room?")
    
    submitted = st.form_submit_button("🔍 Find Matches!")


if submitted:
    # Create new user input row
    new_user = pd.DataFrame([{
        'Wakeup': wakeup,
        'Sleep': sleep,
        'Cleanliness': cleanliness,
        'IntroExtro': intro_extro,
        'StudyTime': study_time,
        'NoiseTolerance': noise_tolerance,
    }])

    new_user_processed = preprocessor.transform(new_user)
    matches = find_top_matches(new_user_processed)

    st.subheader("💌 Your Top Roommate Matches:")
    for name, score in matches:
        emoji = "🌟" if score > 80 else "🧡" if score > 60 else "⚠️"
        st.markdown(f"{emoji} **{name}** — `{score}%` compatible")

# 🎀 Footer
st.markdown(
    "<div style='text-align:center; color:#aaa; margin-top: 2em;'>"
    "Made with 🧠 + 💅 by Kana ✨"
    "</div>",
    """,
    unsafe_allow_html=True
)
