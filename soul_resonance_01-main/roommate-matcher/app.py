import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sqlite3
from datetime import datetime

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
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Load Data ------------------
def load_data():
    # Load static CSV data
    df_csv = pd.read_csv("roommates_mock_data.csv")
    
    # Load submitted data from database
    try:
        conn = sqlite3.connect('roommate_submissions.db')
        df_db = pd.read_sql_query("SELECT * FROM submissions", conn)
        conn.close()
        
        if len(df_db) > 0:
            # Convert database data to match CSV format
            df_db_formatted = pd.DataFrame({
                'Name': df_db['name'],
                'Wakeup': df_db['wakeup'].str.replace('🐓 Early \(6–8 AM\)', 'Early', regex=True)
                                          .str.replace('😴 Mid \(9–11 AM\)', 'Mid', regex=True)
                                          .str.replace('🦥 Late \(12 PM or later\)', 'Late', regex=True),
                'Sleep': df_db['sleep'].str.replace('🌌 Early \(Before 11 PM\)', 'Early', regex=True)
                                       .str.replace('🕰️ Mid \(11 PM – 1 AM\)', 'Mid', regex=True)
                                       .str.replace('🌃 Late \(2 AM or later\)', 'Late', regex=True),
                'Cleanliness': df_db['cleanliness'],
                'IntroExtro': df_db['intro_extro'],
                'StudyTime': df_db['study_time'].str.replace('☀️ Morning', 'Morning', regex=True)
                                                .str.replace('🌙 Night', 'Night', regex=True),
                'NoiseTolerance': df_db['noise_tolerance'],
                'Gender': df_db['gender'].str.replace('👨 Male', 'Male', regex=True)
                                         .str.replace('👩 Female', 'Female', regex=True),
                'IdealRoommate': df_db['looking_for'].fillna('Looking for a compatible roommate') if 'looking_for' in df_db.columns else 'Looking for a compatible roommate'  # Use their description
            })
            
            # Combine CSV and database data
            df_combined = pd.concat([df_csv, df_db_formatted], ignore_index=True)
            return df_combined
        else:
            return df_csv
            
    except Exception as e:
        print(f"Error loading database: {e}")
        return df_csv

df = load_data()

# ------------------ Database Setup ------------------
def init_database():
    conn = sqlite3.connect('roommate_submissions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            gender TEXT NOT NULL,
            looking_for TEXT NOT NULL,
            wakeup TEXT NOT NULL,
            sleep TEXT NOT NULL,
            study_time TEXT NOT NULL,
            cleanliness INTEGER NOT NULL,
            noise_tolerance INTEGER NOT NULL,
            intro_extro REAL NOT NULL,
            submission_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Add looking_for column if it doesn't exist (for existing databases)
    try:
        cursor.execute("ALTER TABLE submissions ADD COLUMN looking_for TEXT DEFAULT '🤝 Any Gender'")
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.commit()
    conn.close()

def save_submission(name, gender, looking_for, wakeup, sleep, study_time, cleanliness, noise_tolerance, intro_extro):
    conn = sqlite3.connect('roommate_submissions.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO submissions (name, gender, looking_for, wakeup, sleep, study_time, cleanliness, noise_tolerance, intro_extro)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (name, gender, looking_for, wakeup, sleep, study_time, cleanliness, noise_tolerance, intro_extro))
    conn.commit()
    conn.close()

def get_all_submissions():
    conn = sqlite3.connect('roommate_submissions.db')
    df = pd.read_sql_query("SELECT * FROM submissions ORDER BY submission_time DESC", conn)
    conn.close()
    return df

def get_submission_count():
    conn = sqlite3.connect('roommate_submissions.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM submissions")
    count = cursor.fetchone()[0]
    conn.close()
    return count

# Initialize database
init_database()

# ------------------ Preprocessing ------------------
features = df[['Wakeup', 'Sleep', 'Cleanliness', 'IntroExtro', 'StudyTime', 'NoiseTolerance']]

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

# Main page header
st.markdown("<h1 style='text-align: center;'>🏠 Hostel Roommate Matcher 🌸</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2em; color: #666;'>Find your perfect roommate match based on lifestyle & personality!</p>", unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.markdown("## 🏨 Tell us about yourself")
st.sidebar.markdown("---")

with st.sidebar:
    with st.form("user_input_form"):
        st.markdown("### 👤 **Your Information**")
        name = st.text_input("📝 Your Name", 
                           placeholder="Enter your full name",
                           help="This will be stored in our database",
                           key="user_name")
        
        gender = st.selectbox("🚻 Your Gender", 
                            ["👨 Male", "👩 Female"],
                            help="Select your gender",
                            key="user_gender")
        
        looking_for = st.text_area("💭 Looking for", 
                                  placeholder="Describe your ideal roommate (e.g., 'quiet and organized', 'fun and social', 'clean and respectful')",
                                  help="Describe what you're looking for in a roommate",
                                  key="looking_for",
                                  max_chars=200)
        
        st.markdown("### ⏰ **Schedule Preferences**")
        wakeup = st.selectbox("🐓 Wake-up Time", 
                            ["🐓 Early (6–8 AM)", "😴 Mid (9–11 AM)", "🦥 Late (12 PM or later)"],
                            help="When do you usually wake up?")
        
        sleep = st.selectbox("🌙 Sleep Time", 
                           ["🌌 Early (Before 11 PM)", "🕰️ Mid (11 PM – 1 AM)", "🌃 Late (2 AM or later)"],
                           help="When do you go to bed?")
        
        study_time = st.selectbox("📚 Study Schedule", 
                                 ["☀️ Morning", "🌙 Night"],
                                 help="When do you prefer to study?",
                                 key="study_schedule")
        
        st.markdown("### 🧹 **Lifestyle Habits**")
        cleanliness = st.slider("🧼 Cleanliness Level", 1, 5, 3,
                               help="1 = Chaotic Goblin, 5 = Neat Freak Supreme 🧽")
        
        noise_tolerance = st.slider("🔊 Noise Tolerance", 1, 5, 3,
                                   help="How much noise can you handle in the room?")
        
        st.markdown("### 🗣️ **Social Energy**")
        intro_extro = st.slider("💬 Social Level", 0.0, 1.0, 0.5,
                               help="0 = Ultra introvert 🙈, 1 = Party Animal 🎉")
        
        st.markdown("---")
        submitted = st.form_submit_button("🔍 Find My Matches!", use_container_width=True)




# Main content area
if not submitted:
    # Landing page content
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h2>🌟 How it works</h2>
        <p style='font-size: 1.1em; color: #666;'>Our smart algorithm matches you with roommates based on:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards in a grid layout
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fff0f5 0%, #ffe4e6 100%); 
                    padding: 1.5rem; border-radius: 15px; margin: 1rem 0;
                    box-shadow: 0 4px 12px rgba(255, 182, 193, 0.15);
                    height: 120px; display: flex; flex-direction: column; justify-content: center;'>
            <h4 style='color: #ff66a1; margin-bottom: 0.5rem; font-size: 1.1rem;'>🕐 Sleep & Wake Schedules</h4>
            <p style='color: #666; margin: 0; font-size: 0.9rem; line-height: 1.4;'>Find someone who matches your daily rhythm and sleep patterns</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fff0f5 0%, #ffe4e6 100%); 
                    padding: 1.5rem; border-radius: 15px; margin: 1rem 0;
                    box-shadow: 0 4px 12px rgba(255, 182, 193, 0.15);
                    height: 120px; display: flex; flex-direction: column; justify-content: center;'>
            <h4 style='color: #ff66a1; margin-bottom: 0.5rem; font-size: 1.1rem;'>🧹 Living Habits</h4>
            <p style='color: #666; margin: 0; font-size: 0.9rem; line-height: 1.4;'>Connect with people who share your cleanliness standards and lifestyle</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fff0f5 0%, #ffe4e6 100%); 
                    padding: 1.5rem; border-radius: 15px; margin: 1rem 0;
                    box-shadow: 0 4px 12px rgba(255, 182, 193, 0.15);
                    height: 120px; display: flex; flex-direction: column; justify-content: center;'>
            <h4 style='color: #ff66a1; margin-bottom: 0.5rem; font-size: 1.1rem;'>🗣️ Social Compatibility</h4>
            <p style='color: #666; margin: 0; font-size: 0.9rem; line-height: 1.4;'>Match your introvert/extrovert energy levels perfectly</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fff0f5 0%, #ffe4e6 100%); 
                    padding: 1.5rem; border-radius: 15px; margin: 1rem 0;
                    box-shadow: 0 4px 12px rgba(255, 182, 193, 0.15);
                    height: 120px; display: flex; flex-direction: column; justify-content: center;'>
            <h4 style='color: #ff66a1; margin-bottom: 0.5rem; font-size: 1.1rem;'>📚 Study Preferences</h4>
            <p style='color: #666; margin: 0; font-size: 0.9rem; line-height: 1.4;'>Find study buddies with similar schedules and habits</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to action
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; background: linear-gradient(135deg, #fce4ec 0%, #f8bbd9 100%); 
                    padding: 1.5rem; border-radius: 15px; margin: 1rem 0;
                    box-shadow: 0 4px 12px rgba(255, 182, 193, 0.2);
                    border: 1px solid #f8bbd9;'>
            <h3 style='color: #ff66a1 !important; margin-bottom: 0.5rem; font-weight: bold;'>🚀 Ready to find your perfect roommate?</h3>
            <p style='color: #666 !important; margin: 0; font-weight: 500;'>👈 Fill out the form in the sidebar to get started!</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Results page
    # Validate that name is provided
    if not name or name.strip() == "":
        st.error("❌ Please enter your name before finding matches!")
        st.stop()
    
    # Validate that looking_for is provided
    if not looking_for or looking_for.strip() == "":
        st.warning("💡 Consider describing what you're looking for in a roommate for better visibility!")
        # Don't stop, just warn - it's optional
    
    # Map form selections to dataset categories
    wakeup_map = {
        "🐓 Early (6–8 AM)": "Early",
        "😴 Mid (9–11 AM)": "Mid", 
        "🦥 Late (12 PM or later)": "Late"
    }
    
    sleep_map = {
        "🌌 Early (Before 11 PM)": "Early",
        "🕰️ Mid (11 PM – 1 AM)": "Mid",
        "🌃 Late (2 AM or later)": "Late"
    }
    
    study_map = {
        "☀️ Morning": "Morning",
        "🌙 Night": "Night"
    }
    
    # Save user submission to database
    try:
        save_submission(
            name.strip(),
            gender,
            looking_for.strip() if looking_for else "",
            wakeup,
            sleep,
            study_time,
            cleanliness,
            noise_tolerance,
            intro_extro
        )
        st.success(f"✅ Welcome {name.strip()}! Your information has been saved.")
        
        # Reload data to include new submission
        df = load_data()
        
        # Update features and preprocessor with new data
        features = df[['Wakeup', 'Sleep', 'Cleanliness', 'IntroExtro', 'StudyTime', 'NoiseTolerance']]
        
        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(), ['Wakeup', 'Sleep', 'StudyTime']),
            ('num', StandardScaler(), ['Cleanliness', 'IntroExtro', 'NoiseTolerance'])
        ])
        
        X = preprocessor.fit_transform(features)
        
    except Exception as e:
        st.warning("⚠️ Could not save your information to database, but proceeding with matching.")
    
    # Create new user input row
    new_user = pd.DataFrame([{
        'Wakeup': wakeup_map[wakeup],
        'Sleep': sleep_map[sleep],
        'Cleanliness': cleanliness,
        'IntroExtro': intro_extro,
        'StudyTime': study_map[study_time],
        'NoiseTolerance': noise_tolerance,
    }])

    new_user_processed = preprocessor.transform(new_user)
    all_matches = find_top_matches(new_user_processed, top_n=20)  # Get more matches for filtering
    
    # Map gender for filtering (back to same-gender matching)
    gender_map = {
        "👨 Male": "Male",
        "👩 Female": "Female"
    }
    
    user_gender = gender_map[gender]
    
    # Filter matches by same gender and exclude current user
    filtered_matches = []
    for match_name, score in all_matches:
        person_gender = df[df['Name'] == match_name]['Gender'].iloc[0]
        # Include only same gender matches and exclude current user
        if match_name.strip() != name.strip() and person_gender == user_gender:
            filtered_matches.append((match_name, score))
    
    # Take top 3 same-gender matches
    matches = filtered_matches[:3]
    
    # Handle case where not enough same-gender matches found
    if len(matches) == 0:
        st.error(f"😔 No {user_gender.lower()} roommates found in our database!")
        st.stop()
    elif len(matches) < 3:
        st.info(f"ℹ️ Only {len(matches)} {user_gender.lower()} match(es) available.")
    
    # Results display
    st.markdown(f"## 💌 {name.strip()}'s Top Roommate Matches")
    st.markdown(f"*Showing {user_gender.lower()} roommates for you*")
    
    # Show user's ideal roommate description if provided
    if looking_for and looking_for.strip():
        st.markdown(f"💭 **You're looking for:** *{looking_for.strip()}*")
    
    # Show total pool size
    total_candidates = len(df[df['Gender'] == user_gender]) - 1  # Exclude current user
    st.markdown(f"*Matching from a pool of {total_candidates} {user_gender.lower()} candidates*")
    
    st.markdown("---")
    
    # Display each match in a nice card format
    for i, (name, score) in enumerate(matches, 1):
        # Get detailed info for this person
        person_info = df[df['Name'] == name].iloc[0]
        
        # Create columns for better layout
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Match ranking and score
            if score > 80:
                st.markdown(f"### 🥇 #{i}")
                st.success(f"**{score}%** Match!")
            elif score > 60:
                st.markdown(f"### 🥈 #{i}")
                st.info(f"**{score}%** Match!")
            else:
                st.markdown(f"### 🥉 #{i}")
                st.warning(f"**{score}%** Match!")
        
        with col2:
            # Check if this is a submitted candidate or from original data
            original_names = pd.read_csv("roommates_mock_data.csv")['Name'].tolist()
            is_submitted = name not in original_names
            
            if is_submitted:
                st.markdown(f"### 👤 **{name}** 🆕")
                st.caption("*New candidate from submissions*")
            else:
                st.markdown(f"### 👤 **{name}**")
            
            # Create info cards
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.markdown("**Schedule:**")
                st.write(f"🌅 Wakes up: {person_info['Wakeup']}")
                st.write(f"🌙 Sleeps: {person_info['Sleep']}")
                st.write(f"📚 Studies: {person_info['StudyTime']}")
                
            with info_col2:
                st.markdown("**Lifestyle:**")
                st.write(f"🧼 Cleanliness: {person_info['Cleanliness']}/5")
                st.write(f"🔊 Noise tolerance: {person_info['NoiseTolerance']}/5")
                social_level = "Introvert" if person_info['IntroExtro'] < 0.5 else "Extrovert"
                st.write(f"💬 Social: {social_level}")
                # Handle gender display with fallback
                if 'Gender' in person_info and pd.notna(person_info['Gender']):
                    gender_emoji = "👨" if person_info['Gender'] == 'Male' else "👩"
                    st.write(f"{gender_emoji} Gender: {person_info['Gender']}")
                else:
                    st.write("👤 Gender: Not specified")
            
            # Ideal roommate description
            if pd.notna(person_info['IdealRoommate']):
                st.markdown(f"**💭 Looking for:** *{person_info['IdealRoommate']}*")
        
        st.markdown("---")



# 🎀 Footer
st.markdown(
    "<div style='text-align:center; color:#aaa; margin-top: 2em;'>"
    "🫐 Built to prevent rommate trauma :D 🫐"
    "</div>",
    unsafe_allow_html=True
)
